from utils.client_tools import annotate_frame
from utils.processor import send_batch, setup_camera, batch_inference_worker, start_workers, send_frame_to_queue, run_inference_loop, cleanup_workers
import cv2
from multiprocessing import Process, Queue
import time
import requests
from requests.adapters import HTTPAdapter
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def main():
    """Main function to setup and run batch inference system"""
    # Configuration
    endpoint = "https://c21738f0e505.ngrok-free.app" + "/infer/batch"
    camera_index = 0
    target_fps = 5
    batch_size = 4
    num_workers = 1
    
    # Create queues
    max_queue_size = batch_size * num_workers * 2
    input_queue = Queue(maxsize=max_queue_size)
    output_queue = Queue(maxsize=max_queue_size)
    
    # Start workers
    workers = start_workers(num_workers, input_queue, output_queue, endpoint, batch_size)
    
    # Setup camera
    cap = setup_camera(camera_index)
    
    print(f"Started with {num_workers} workers")
    print(f"Target FPS: {target_fps}, Batch Size: {batch_size}")
    
    try:
        run_inference_loop(cap, input_queue, output_queue, target_fps, batch_size)
    finally:
        cleanup_workers(workers, input_queue)
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    main()