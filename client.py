from utils.processor import infer_and_annotate
import cv2
import threading
from multiprocessing import Process, Queue
import time

def inference_worker(input_queue, output_queue, endpoint, fps):
    """Background thread for processing frames"""
    min_interval = 1 / fps
    last_process_time = 0
    last_frame = None
    
    while True:
        if not input_queue.empty():
            frame = input_queue.get()
            if frame is None:  # Poison pill to stop thread
                break
            
            current_time = time.time()
            if current_time - last_process_time >= min_interval:
                annotated = infer_and_annotate(frame, min_interval=min_interval, endpoint=endpoint)
                last_process_time = current_time
                last_frame = annotated
            else:
                # Use last annotated frame if throttling
                if last_frame is not None:
                    annotated = last_frame
                else:
                    annotated = frame
            
            # Keep only latest result
            while not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except:
                    break
            output_queue.put(annotated)

if __name__ == "__main__":
    endpoint = "https://ecd910d4202a.ngrok-free.app/infer"  # Replace with your inference server URL
    camera_index = 0
    fps = 10
    max_frames_in_queue = 2

    input_queue = Queue(maxsize=max_frames_in_queue)  # Limit queue size
    output_queue = Queue(maxsize=max_frames_in_queue)

    # Start inference worker thread
    worker = threading.Thread(
        target=inference_worker,
        args=(input_queue, output_queue, endpoint, fps),
        daemon=True
    )
    worker.start()
    
    cap = cv2.VideoCapture(camera_index)
    last_annotated = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Send frame to inference queue (non-blocking)
        if not input_queue.full():
            input_queue.put(frame)
        
        # Get latest annotated frame if available
        if not output_queue.empty():
            last_annotated = output_queue.get()
        
        # Display latest annotated frame or original
        display_frame = last_annotated if last_annotated is not None else frame
        
        cv2.imshow("Annotated Video Stream", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    input_queue.put(None)  # Stop worker
    worker.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()