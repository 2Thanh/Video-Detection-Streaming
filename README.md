# 🎧 Video Detection Streaming

A **high-performance real-time video inference system** that streams video from a **local camera** and performs **object detection** using a **remote YOLOv8 API** hosted on **Google Colab**.
The system uses a **threaded queue-based architecture** for smooth, non-blocking video processing with stable frame rates and responsive display.

---

## ✨ Features

* 🎥 **Local camera streaming** via OpenCV
* ☁️ **Remote inference** through FastAPI endpoint (e.g., Google Colab or server)
* 🚀 **Non-blocking processing** using threaded queue design
* ⚡ **FPS throttling** for adjustable inference speed
* 🎯 **YOLOv8 object detection** with customizable confidence threshold
* 🧠 **Smooth visualization** with zero UI freezing
* 🔄 **Automatic frame skipping** to maintain real-time performance

---

## 🧩 Requirements

* Python **3.8+**
* OpenCV (`cv2`)
* `ultralytics`
* `processor.py` (must implement `infer_and_annotate()`)

Install dependencies:

```bash
pip install opencv-python ultralytics fastapi uvicorn
```

---

## 🚀 Usage

### 🖥️ On Google Colab (Server Side)

1. **Install dependencies:**

```bash
pip install ultralytics fastapi uvicorn
```

2. **Install ngrok for public access:**

```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
&& echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" | sudo tee /etc/apt/sources.list.d/ngrok.list \
&& sudo apt update && sudo apt install ngrok

ngrok config add-authtoken <YOUR_NGROK_TOKEN>
```

3. **Run the FastAPI YOLOv8 server:**

```bash
uvicorn server:app --reload
```

4. **Expose your local API to the internet:**

```bash
ngrok http 8000
```

You’ll receive a public URL like:

```
https://<random>.ngrok-free.app
```

Use that URL as your **remote inference endpoint** in the client.

---

### 💻 On Local Machine (Client Side)

1. Set your remote server URL inside `client.py`, for example:

```python
REMOTE_API = "https://<random>.ngrok-free.app/infer"
```

2. Run the client:

```bash
python Office/client.py
```

3. Press **`q`** to quit the video stream.

---

## ⚙️ Customization

* 🕒 Adjust processing frame rate: modify the `fps` value in `client.py`
* 🎨 Customize annotation logic: edit `infer_and_annotate()` in `processor.py`
* 🎚️ Change confidence threshold: update `conf` parameter in the API call

---

## 🧠 Architecture Overview

```
+---------------------+
|   Local Camera      |
+----------+----------+
           |
           v
+---------------------+
|  Frame Queue Thread |
+----------+----------+
           |
           v
+---------------------+
|  Remote Inference   |  (via FastAPI + YOLOv8 on Colab)
+----------+----------+
           |
           v
+---------------------+
|  Display Annotated  |
|      Frames         |
+---------------------+
```

---

## 🎥 Demo

<img src="source/demo.gif" width="640" alt="YOLOv8 Detection Demo">
---

## 📄 License

**MIT License** — feel free to use, modify, and distribute this project.

