from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Union, Tuple

class YoloInfer:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def load_model(self, model_path):
        self.model = YOLO(model_path)

    def draw_box(self, image, boxes, labels, scores):
        """Draw bounding boxes on a single image"""
        for box, cls, conf in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            color = (0, 255, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            
            # Draw label background for better visibility
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(image, (int(x1), int(y1) - text_height - 10),
                         (int(x1) + text_width, int(y1)), color, -1)
            cv2.putText(image, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return image

    def inference(self, image, conf_threshold):
        """Single image inference (backward compatible)"""
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.model(bgr_image, conf=conf_threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        output_bgr = self.draw_box(bgr_image.copy(), boxes, labels, scores)
        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), boxes, labels, scores

    def inference_batch(self, images: List[np.ndarray], conf_threshold: float = 0.5):
        """
        Batch inference for multiple images - uses GPU batch processing
        
        Args:
            images: List of images in RGB format
            conf_threshold: Confidence threshold
            
        Returns:
            List of tuples (annotated_image, boxes, labels, scores)
        """
        # Convert all images to BGR
        bgr_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        
        # TRUE BATCH INFERENCE - Single GPU call for all images
        results = self.model(bgr_images, conf=conf_threshold)
        
        # Process results for each image
        batch_results = []
        for idx, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            # Draw boxes on the image
            output_bgr = self.draw_box(bgr_images[idx].copy(), boxes, labels, scores)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            
            batch_results.append((output_rgb, boxes, labels, scores))
        
        return batch_results

    def inference_batch_raw(self, images: List[np.ndarray], conf_threshold: float = 0.5):
        """
        Batch inference returning only detection data (no drawing)
        Faster when you don't need annotated images
        
        Args:
            images: List of images in RGB format
            conf_threshold: Confidence threshold
            
        Returns:
            List of tuples (boxes, labels, scores)
        """
        # Convert all images to BGR
        bgr_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        
        # Batch inference
        results = self.model(bgr_images, conf=conf_threshold)
        
        # Extract detection data only
        batch_results = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            batch_results.append((boxes, labels, scores))
        
        return batch_results

    def inference_batch_dict(self, images: List[np.ndarray], conf_threshold: float = 0.5):
        """
        Batch inference returning structured detection data
        
        Args:
            images: List of images in RGB format
            conf_threshold: Confidence threshold
            
        Returns:
            List of detection dictionaries
        """
        # Convert all images to BGR
        bgr_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        
        # Batch inference
        results = self.model(bgr_images, conf=conf_threshold)
        
        # Format as structured data
        batch_results = []
        for result in results:
            detections = []
            if len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                
                for box, label, score in zip(boxes, labels, scores):
                    detections.append({
                        "bbox": box.tolist(),
                        "label": self.model.names[int(label)],
                        "class_id": int(label),
                        "confidence": float(score)
                    })
            
            batch_results.append(detections)
        
        return batch_results