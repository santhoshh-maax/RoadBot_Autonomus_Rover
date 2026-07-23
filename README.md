# 🚗 Road Bot: AI-Powered Autonomous Pothole Detection and Repair System [![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-db61a2?style=flat&logo=github-sponsors)](https://github.com/sponsors/santhoshh-maax)

An AI-powered autonomous rover that detects potholes using **YOLOv8**, measures their dimensions using the **Intel RealSense D455f Depth Camera**, and is designed to automate road repair through an intelligent material dispensing mechanism.

This project combines **Computer Vision**, **Edge AI**, **Depth Sensing**, and **Robotics** to create a smart solution for road maintenance.

---

# 📖 Project Overview

Road Bot is designed to reduce manual road inspections by automatically detecting potholes and estimating their dimensions in real time. The system uses an Intel RealSense RGB-D camera for accurate depth measurement and a custom-trained YOLOv8 model for pothole detection. Once detected, the rover calculates the pothole volume and can be integrated with an automated repair mechanism.

---

# ✨ Features

- 🤖 Autonomous Rover Navigation
- 🕳️ Real-Time Pothole Detection
- 📷 Intel RealSense RGB-D Depth Camera
- 📏 3D Pothole Measurement
- 🧠 YOLOv8 Object Detection
- 📍 RTK-GNSS Location Mapping
- ⚙️ ESP32-Based Rover Control
- 📊 CSV Measurement Logging
- 🚧 Automatic Repair System (Prototype)
- ⚡ Edge AI using NVIDIA Jetson Nano

---

# 🖼️ System Architecture

<p align="center">
    <img src="block diagram.png" width="900">
</p>

---

# 🛠️ Hardware Components

- NVIDIA Jetson Nano
- Intel RealSense D455f Depth Camera
- ESP32 Development Board
- RTK-GNSS Module
- 4WD Rover Chassis
- Motor Driver
- DC Gear Motors
- Screw Conveyor Mechanism
- Lead Acid Battery
- Power Distribution Module

---

# 💻 Software Used

- Python 3.10
- Ultralytics YOLOv8
- OpenCV
- PyRealSense2
- Open3D
- NumPy
- Roboflow
- Arduino IDE (ESP32)

---

# 📂 Repository Structure

```text
Road-Bot/
│
├── bluetoothcar/
├── rovercontroller/
├── dataset/
├── PATH HOLE DETECTION.V1.yolov8/
├── model outputs/
├── references/
├── runs/
│
├── main.py
├── rover_final.py
├── rover_laptop.py
│
├── block diagram.png
├── road.jpg
├── hole_measurements.csv
│
├── how to run Yolov8 Model.docx
├── python3.10 installation with realsense.docx
├── connect jetson with esp32.docx
│
└── README.md
```

---

# ⚙️ Working Principle

1. The Intel RealSense D455f captures RGB and Depth frames.
2. YOLOv8 detects potholes from the RGB image.
3. The depth image verifies the pothole and measures its dimensions.
4. The Jetson Nano calculates the pothole volume.
5. Commands are sent to the ESP32 to control rover movement.
6. GPS coordinates and measurements are stored for future analysis.
7. The repair mechanism can dispense the required amount of repair material.

---

# 🧠 AI Model

The pothole detection model was trained using a custom dataset collected from real road conditions.

### Dataset

- 400+ Real-World Images
- Annotated using Roboflow
- YOLOv8 Format

### Training Command

```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16
```

### Output

```
best.pt
```

---

# 📊 Output

The system can generate:

- ✅ Pothole Detection
- ✅ Bounding Box
- ✅ Length, Width & Depth Measurement
- ✅ Volume Estimation
- ✅ GPS Coordinates
- ✅ CSV Measurement Report

---

# 📄 Documentation

This repository includes:

- 📘 YOLOv8 Model Execution Guide
- 📘 Python 3.10 & RealSense Installation Guide
- 📘 Jetson Nano ↔ ESP32 Communication Guide
- 📊 Hole Measurement CSV
- 🖼️ Block Diagram
- 📷 Reference Images

---

# ⚠️ Important Note

> **This project is designed to work with the Intel RealSense D455f Depth Camera.**

The Python programs in this repository use the **PyRealSense2** library to capture synchronized RGB and depth frames. Therefore, the complete pothole detection and measurement pipeline **cannot be executed using a normal USB webcam**.

### Don't have an Intel RealSense Camera?

You can still test the trained **YOLOv8 model** using images or videos.

Please refer to:

📄 **how to run Yolov8 Model.docx**

This document explains:

- Python installation
- Required libraries
- Model setup
- Running inference on images
- Running inference on videos

---

# 🚀 Future Improvements

- SLAM-Based Autonomous Navigation
- TensorRT Model Optimization
- Cloud Dashboard
- Automatic Concrete Mixing
- Multi-Class Road Damage Detection
- Mobile Monitoring Application
- Fleet Management System

---

# 👨‍💻 Author

**Santhosh P**

B.E. Computer Science and Engineering

Mount Zion College of Engineering and Technology

---

# ⭐ Support

👉 **[Sponsor me on GitHub](https://github.com/sponsors/santhoshh-maax)**

If you found this project useful, please consider giving this repository a **⭐ Star**.

---

# 📜 License

This project is developed for educational, research, and prototype development purposes.
