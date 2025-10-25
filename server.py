from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from utils.tools import YoloInfer

# Initialize YOLO inference
yolo = YoloInfer("yolov8n.pt")

app = FastAPI()

@app.post("/infer")
async def infer_image(file: UploadFile = File(...), conf: float = Form(0.5)):
    # Read image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    output_image, boxes, labels, scores = yolo.inference(image, conf)

    # Prepare response
    detections = []
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        detections.append({
            "bbox": box,      # [x1, y1, x2, y2]
            "label": yolo.model.names[int(label)],
            "confidence": round(score, 3)
        })

    return JSONResponse(content={"detections": detections})