from fastrtc import Stream
import numpy as np
import requests
import cv2
import io
import time

def process_output(output, image):
    # {'detections': [{'bbox': [336.0950927734375, 48.94780349731445, 740.3195190429688, 715.763427734375], 'label': 'person', 'confidence': 0.929}]}
    detections = output.get("detections", [])
    for detection in detections:
        bbox = detection.get("bbox", [])
        label = detection.get("label", "")
        confidence = detection.get("confidence", 0)
        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            image,
            f"{label}: {confidence:.2f}",
            (int(bbox[0]), int(bbox[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    return image

def request_inference(img, conf_threshold=0.5, endpoint=None):
    # Encode as JPEG (to send over HTTP)
    _, img_encoded = cv2.imencode(".jpg", img)

    # Send POST request with the encoded image
    files = {"file": ("image.jpg", io.BytesIO(img_encoded.tobytes()), "image/jpeg")}
    data = {"conf": str(conf_threshold)}

    response = requests.post(endpoint, files=files, data=data)
    return response.json()


def infer_and_annotate(image, min_interval, last_process_time=0, endpoint=None):
    current_time = time.time()
    
    if current_time - last_process_time < min_interval:
        return image  # Return original without processing
    
    last_process_time = current_time
    output = request_inference(image, conf_threshold=0.5, endpoint=endpoint)
    annotated_image = process_output(output, image)
    return annotated_image