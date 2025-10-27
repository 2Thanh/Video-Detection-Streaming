from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from utils.server_tools import YoloInfer
from typing import List
import asyncio

# Initialize YOLO inference
yolo = YoloInfer("yolov8n.pt")

app = FastAPI()

@app.post("/infer")
async def infer_image(file: UploadFile = File(...), conf: float = Form(0.5)):
    """Single image inference"""
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
            "bbox": box,
            "label": yolo.model.names[int(label)],
            "confidence": round(score, 3)
        })

    return JSONResponse(content={"detections": detections})

@app.post("/infer/batch")
async def infer_images_batch(
    files: List[UploadFile] = File(...),
    conf: float = Form(0.5)
):
    """GPU batch inference - MUCH faster than sequential"""
    # Read all files concurrently
    file_contents = await asyncio.gather(*[file.read() for file in files])
    
    # Decode all images
    images = []
    filenames = []
    for idx, contents in enumerate(file_contents):
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        filenames.append(files[idx].filename)
    
    # TRUE BATCH INFERENCE - Single GPU call
    batch_detections = yolo.inference_batch_dict(images, conf)
    
    # Format response
    results = []
    for idx, detections in enumerate(batch_detections):
        results.append({
            "filename": filenames[idx],
            "detections": detections
        })
    
    return JSONResponse(content={
        "total_images": len(results),
        "results": results
    })