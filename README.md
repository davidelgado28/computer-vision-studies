# Self-Contained OpenCV Object Detector

A high-performance, single-file real-time object detection system built with Python and OpenCV (`cv2.dnn`), designed with a senior-level architecture. It features automatic model downloading, live webcam streaming, and an interactive snapshot utility.

---

## Features

* **Zero External Dependencies Setup**: Automatically downloads the required MobileNet-SSD Caffe weights and configuration files on first run.
* **Single-File Architecture**: Clean, modular, and self-contained within a single Python script (`main.py`).
* **Real-Time Inference**: Processes live video feeds from webcams with integrated dynamic FPS tracking.
* **Instant Snapshot Utility**: Press **`s`** during runtime to capture and save annotated frames into an `output/` directory with unique timestamps.
* **Robust Resource Management**: Safe lifecycle management ensuring camera streams and windows are properly released upon exit or error.

---

## Prerequisites

Make sure you have Python 3.8+ installed. You will only need the core computer vision and numerical libraries:

```bash
pip install opencv-python numpy
