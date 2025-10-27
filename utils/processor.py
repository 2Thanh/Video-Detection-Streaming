from fastrtc import Stream
import numpy as np
import requests
import cv2
import io
import time
from multiprocessing import Process, Queue
from utils.client_tools import annotate_frame
from requests.adapters import HTTPAdapter
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def send_batch(batch, endpoint, output_queue, session):
    """Send batch of frames for inference"""
    try:
        files = []
        for frame_id, frame in batch:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            files.append(('files', (f'frame_{frame_id}.jpg', buffer.tobytes(), 'image/jpeg')))
        
        response = session.post(
            endpoint,
            files=files,
            data={'conf': 0.5},
            timeout=10,
            verify=False
        )
        
        if response.status_code == 200:
            results = response.json()['results']
            
            # Send back annotated frames
            for i, result in enumerate(results):
                frame_id, frame = batch[i]
                annotated = annotate_frame(frame.copy(), result['detections'])
                output_queue.put((frame_id, annotated))
        else:
            print(f"Batch request failed: {response.status_code}")
            # Return original frames
            for frame_id, frame in batch:
                output_queue.put((frame_id, frame))
                
    except Exception as e:
        print(f"Batch send error: {e}")
        # Return original frames on error
        for frame_id, frame in batch:
            output_queue.put((frame_id, frame))


def batch_inference_worker(input_queue, output_queue, endpoint, batch_size, worker_id):
    """Worker process that collects frames into batches and sends for inference"""
    print(f"Worker {worker_id} started with batch_size={batch_size}")
    
    # Setup session for this worker
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=5, pool_maxsize=10, max_retries=2)
    session.mount("https://", adapter)
    
    batch = []
    batch_timeout = 0.1  # Max time to wait for batch to fill
    last_batch_time = time.time()
    
    while True:
        try:
            # Try to get a frame (with timeout)
            try:
                item = input_queue.get(timeout=0.05)
                
                # Check for poison pill
                if item is None:
                    # Process remaining batch before exiting
                    if batch:
                        send_batch(batch, endpoint, output_queue, session)
                    session.close()
                    print(f"Worker {worker_id} stopped")
                    return
                
                batch.append(item)
            except:
                pass  # Timeout, continue to check if we should send batch
            
            current_time = time.time()
            should_send = (
                len(batch) >= batch_size or
                (batch and current_time - last_batch_time >= batch_timeout)
            )
            
            if should_send:
                send_batch(batch, endpoint, output_queue, session)
                batch = []
                last_batch_time = current_time
                
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            time.sleep(0.1)


def start_workers(num_workers, input_queue, output_queue, endpoint, batch_size):
    """Start worker processes for batch inference"""
    workers = []
    for i in range(num_workers):
        worker = Process(
            target=batch_inference_worker,
            args=(input_queue, output_queue, endpoint, batch_size, i),
            daemon=True
        )
        worker.start()
        workers.append(worker)
    return workers


def setup_camera(camera_index):
    """Initialize camera with optimal settings"""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def send_frame_to_queue(input_queue, frame_id, frame, current_time, last_capture_time, frame_interval):
    """Send frame to worker queue if enough time has passed"""
    if current_time - last_capture_time >= frame_interval:
        if not input_queue.full():
            input_queue.put((frame_id, frame.copy()))
            return frame_id + 1, current_time
    return frame_id, last_capture_time


def collect_results(output_queue, results_buffer):
    """Collect all available results from workers"""
    while not output_queue.empty():
        try:
            result_id, annotated = output_queue.get_nowait()
            results_buffer[result_id] = annotated
        except:
            break


def get_next_display_frame(results_buffer, next_display_id, last_annotated, batch_size):
    """Get next sequential frame from buffer and cleanup old frames"""
    if next_display_id in results_buffer:
        last_annotated = results_buffer.pop(next_display_id)
        next_display_id += 1
        
        # Clean up old buffered frames
        if len(results_buffer) > batch_size * 4:
            old_ids = [fid for fid in results_buffer.keys() if fid < next_display_id]
            for old_id in old_ids:
                del results_buffer[old_id]
    
    return next_display_id, last_annotated


def calculate_fps(fps_counter, fps_start, current_time):
    """Calculate FPS and determine if it should be updated"""
    fps_counter += 1
    if current_time - fps_start >= 1.0:
        actual_fps = fps_counter / (current_time - fps_start)
        return 0, current_time, actual_fps
    return fps_counter, fps_start, None


def display_frame_with_fps(frame, fps_value=None):
    """Display frame with optional FPS overlay"""
    if fps_value is not None:
        cv2.putText(frame, f"FPS: {fps_value:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Batch Inference", frame)


def cleanup_workers(workers, input_queue):
    """Gracefully shutdown worker processes"""
    print("Shutting down...")
    
    # Send poison pills to workers
    for _ in range(len(workers)):
        input_queue.put(None)
    
    # Wait for workers to finish
    for worker in workers:
        worker.join(timeout=2)
        if worker.is_alive():
            worker.terminate()


def run_inference_loop(cap, input_queue, output_queue, target_fps, batch_size):
    """Main inference loop that captures, processes, and displays frames"""
    # Frame tracking
    frame_id = 0
    results_buffer = {}
    next_display_id = 0
    last_annotated = None
    
    # FPS control
    frame_interval = 1.0 / target_fps
    last_capture_time = 0
    
    # FPS monitoring
    fps_counter = 0
    fps_start = time.time()
    current_fps = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # Send frame to worker queue
        frame_id, last_capture_time = send_frame_to_queue(
            input_queue, frame_id, frame, current_time, last_capture_time, frame_interval
        )
        
        # Collect results from workers
        collect_results(output_queue, results_buffer)
        
        # Get next sequential frame
        next_display_id, last_annotated = get_next_display_frame(
            results_buffer, next_display_id, last_annotated, batch_size
        )
        
        # Determine which frame to display
        display_frame = last_annotated if last_annotated is not None else frame
        
        # Calculate FPS
        fps_counter, fps_start, new_fps = calculate_fps(fps_counter, fps_start, current_time)
        if new_fps is not None:
            current_fps = new_fps
        
        # Display frame with FPS
        display_frame_with_fps(display_frame, current_fps)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
