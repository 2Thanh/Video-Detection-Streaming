from ultralytics import YOLO
import cv2
import numpy as np

class YoloInfer:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def load_model(self, model_path):
        self.model = YOLO(model_path)

    def draw_box(self, image, boxes, labels, scores):
        for box, cls, conf in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            color = (0, 255, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image

    def inference(self, image, conf_threshold):
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.model(bgr_image, conf=conf_threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        output_bgr = self.draw_box(bgr_image.copy(), boxes, labels, scores)
        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), boxes, labels, scores