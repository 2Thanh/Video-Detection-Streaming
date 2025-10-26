from utils.processor import infer_and_annotate
import cv2
from multiprocessing import Process, Queue
import time
import requests
from requests.adapters import HTTPAdapter
import numpy as np

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def inference_worker_batch(input_queue, output_queue, endpoint, worker_id, batch_size=2):
    """Worker with smaller batches and better connection handling"""
    print(f"Worker {worker_id} started (batch_size={batch_size})")
    
    # Create session with connection pooling
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=5,
        pool_maxsize=10,
        max_retries=2
    )
    session.mount("https://", adapter)
    
    batch_timeout = 0.1  # Increased timeout for collection
    
    while True:
        try:
            batch = []
            start_collect = time.time()
            
            while len(batch) < batch_size:
                remaining_time = batch_timeout - (time.time() - start_collect)
                if remaining_time <= 0 and batch:
                    break
                
                try:
                    item = input_queue.get(timeout=max(0.01, remaining_time))
                    if item is None:
                        if batch:
                            process_batch_safe(batch, endpoint, output_queue, session)
                        session.close()
                        return
                    batch.append(item)
                except:
                    break
            
            if batch:
                process_batch_safe(batch, endpoint, output_queue, session)
            else:
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            time.sleep(0.1)  # Back off on error
            continue
    
    print(f"Worker {worker_id} stopped")

def process_batch_safe(batch, endpoint, output_queue, session):
    """Process batch with comprehensive error handling"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            files = []
            frame_ids = []
            
            for frame_id, frame in batch:
                frame_ids.append(frame_id)
                # Use higher quality to reduce corruption issues
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                files.append(('files', (f'frame_{frame_id}.jpg', buffer.tobytes(), 'image/jpeg')))
            
            response = session.post(
                endpoint.replace('/infer', '/infer/batch'),
                files=files,
                data={'conf': 0.5},
                timeout=15,  # Generous timeout
                verify=False,
                headers={'Connection': 'close'}  # Don't keep connection alive
            )
            
            if response.status_code == 200:
                results = response.json()['results']
                
                for i, result in enumerate(results):
                    frame_id = frame_ids[i]
                    _, frame = batch[i]
                    annotated = annotate_frame(frame.copy(), result['detections'])
                    output_queue.put((frame_id, annotated))
                return  # Success!
            else:
                print(f"Batch request failed: {response.status_code}")
                
        except requests.exceptions.SSLError as e:
            print(f"SSL Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Wait before retry
                continue
        except Exception as e:
            print(f"Batch processing error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
    
    # All retries failed, return original frames
    print(f"All retries failed for batch of {len(batch)} frames")
    for frame_id, frame in batch:
        output_queue.put((frame_id, frame))


def process_batch(batch, endpoint, output_queue):
    """Send batch of frames to server for GPU batch inference"""
    try:
        # Prepare batch request
        files = []
        frame_ids = []
        
        for frame_id, frame in batch:
            frame_ids.append(frame_id)
            # Encode frame to JPEG (lower quality for speed)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            files.append(('files', (f'frame_{frame_id}.jpg', buffer.tobytes(), 'image/jpeg')))
        
        # Send batch request
        response = requests.post(
            endpoint.replace('/infer', '/infer/batch'),
            files=files,
            data={'conf': 0.5},
            timeout=5
        )
        
        if response.status_code == 200:
            results = response.json()['results']
            
            # Match results back to frames
            for i, result in enumerate(results):
                frame_id = frame_ids[i]
                _, frame = batch[i]
                
                # Annotate frame with detections
                annotated = annotate_frame(frame.copy(), result['detections'])
                output_queue.put((frame_id, annotated))
        else:
            print(f"Batch request failed: {response.status_code}")
            # Return original frames
            for frame_id, frame in batch:
                output_queue.put((frame_id, frame))
                
    except Exception as e:
        print(f"Batch processing error: {e}")
        # Return original frames on error
        for frame_id, frame in batch:
            output_queue.put((frame_id, frame))

def annotate_frame(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for det in detections:
        bbox = det['bbox']
        label = det['label']
        conf = det['confidence']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = f"{label}: {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

if __name__ == "__main__":
    endpoint = "https://9c9fb751298f.ngrok-free.app/infer"
    camera_index = 0
    fps = 15  # Increased FPS since batch inference is faster
    num_workers = 1  # Fewer workers needed with batch processing
    batch_size = 4  # Larger batches for better GPU utilization
    max_queue_size = num_workers * batch_size * 2
    
    input_queue = Queue(maxsize=max_queue_size)
    output_queue = Queue(maxsize=max_queue_size)
    
    # Start multiple worker processes
    workers = []
    for i in range(num_workers):
        worker = Process(
            target=inference_worker_batch,
            args=(input_queue, output_queue, endpoint, i, batch_size),
            daemon=True
        )
        worker.start()
        workers.append(worker)
    
    cap = cv2.VideoCapture(camera_index)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_id = 0
    last_annotated = None
    frame_buffer = {}
    next_display_id = 0
    
    # FPS throttling
    min_interval = 1 / fps
    last_send_time = 0
    
    # Performance monitoring
    fps_counter = 0
    fps_start_time = time.time()
    
    print(f"Started with {num_workers} workers, batch_size={batch_size}")
    print(f"Target FPS: {fps}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            # Throttle frame sending based on FPS
            if current_time - last_send_time >= min_interval:
                if not input_queue.full():
                    try:
                        input_queue.put_nowait((frame_id, frame.copy()))
                        frame_id += 1
                        last_send_time = current_time
                    except:
                        pass
            
            # Collect all available results
            while not output_queue.empty():
                try:
                    result_id, annotated = output_queue.get_nowait()
                    frame_buffer[result_id] = annotated
                except:
                    break
            
            # Get the next sequential frame if available
            if next_display_id in frame_buffer:
                last_annotated = frame_buffer.pop(next_display_id)
                next_display_id += 1
                
                # Clean up old buffered frames (keep buffer small)
                if len(frame_buffer) > batch_size * 2:
                    # Skip to latest available frame to reduce lag
                    available_ids = sorted(frame_buffer.keys())
                    if available_ids:
                        next_display_id = available_ids[-1]
                        last_annotated = frame_buffer[next_display_id]
                        frame_buffer = {}
            
            # Display latest annotated frame or original
            display_frame = last_annotated if last_annotated is not None else frame
            
            # Calculate and display FPS
            fps_counter += 1
            if current_time - fps_start_time >= 1.0:
                actual_fps = fps_counter / (current_time - fps_start_time)
                cv2.putText(display_frame, f"FPS: {actual_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                fps_counter = 0
                fps_start_time = current_time
            
            cv2.imshow("Annotated Video Stream", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        print("Shutting down...")
        for _ in range(num_workers):
            input_queue.put(None)
        
        for worker in workers:
            worker.join(timeout=3)
            if worker.is_alive():
                worker.terminate()
        
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")