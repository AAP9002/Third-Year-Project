# 📘 Third Year Project – Road Bend Classification

**Project Report**: [Overleaf Link](https://www.overleaf.com/read/xwtwzrkkstnt#bc3052)

**Presentation**: [Video](https://youtu.be/B2moIPvdwpo)

**Presentation Slides**: [Reveal.js](https://aap9002.github.io/Road-Bend-Classification-Presentation/)

**Project Log**: [Notion Link](https://alansnotes.notion.site/Third-Year-Project-Logs-13d57d67a5bf805a8a6ff1cfe588fc89?pvs=4)

---

### 🎯 Overview

This project aims to enhance **Advanced Driver Assistance Systems (ADAS)** by predicting the severity and direction of upcoming road bends using dashcam footage. It introduces a **bespoke UK road dataset** and proposes a deep learning-based classification system that integrates optical flow and RGB representations from stereo dashcam video.

---

### 💡 Ideas Explored

This research investigates the effectiveness of human-inspired perception cues in machine-based road understanding. Specifically:

- 🧠 **H1**: Deep Neural Networks (DNNs) can generalise the task of road bend severity and direction classification effectively from video input.
- 👁️ **H2**: Optical flow representations, inspired by human visual motion processing, improve classification performance compared to static RGB imagery.
- 🎯 **H3**: Focused regions around the Road Vanishing Point (R-VP), emulating human gaze, result in better or more efficient classification performance.

Each investigation is evaluated using separate datasets (RGB vs. Optical Flow, Wide vs. Narrow views) and a consistent (2+1)D CNN architecture.

---

### 📊 Dataset & Resources

| Resource                        | Link |
|-------------------------------|------|
| Final Dataset                 | [UK-Road-Bend-Classification](https://huggingface.co/datasets/aap9002/UK-Road-Bend-Classification) |
| Trained Models                | [RGB & Optical Flow Models](https://huggingface.co/aap9002/RGB_Optic_Flow_Bend_Classification) |
| Raw Dashcam Videos            | [UK-Road-DashCam](https://huggingface.co/datasets/aap9002/UK-Road-DashCam) |
| Component Testing Dataset     | [Stereo-Road-Curvature-Dashcam](https://huggingface.co/datasets/aap9002/Stereo-Road-Curvature-Dashcam) |
| Source Code                   | [GitHub Repo](https://github.com/AAP9002/Third-Year-Project) |
| Calibration Files             | [Camera Calibration](https://huggingface.co/datasets/aap9002/Stereo-Road-Curvature-Dashcam/tree/main/camera_calibration) |

---

### 📈 Key Results

- **RGB Wide View** outperformed other configurations, achieving **73.78% accuracy**.
- **Optical Flow Narrow View** had **55.55% accuracy**, supporting human-based motion field inspired representation.
- Extensive evaluation with F1 scores and confusion matrices confirms the efficacy of the models under varied driving conditions.

---

### 🔍 Future Work

- Integrate **stereo depth estimation** to improve bend understanding.
- Use **microcontroller-driven ego-motion correction**.
- Experiment with **transformer-based or hybrid DNN architectures**.

---

### ⚙️ Processing Pipeline

The processing pipeline is composed of several modular stages:

1. **Bend Detection & Labelling**  
   - Based on GPS heading change over distance-normalised segments.
   - Uses NMEA-formatted GPS embedded in video.
   
2. **R-VP Estimation**  
   - Road Vanishing Point detection using optical flow and feature tracking.
   
3. **Optical Flow Analysis**  
   - Dense optical flow and rotational homography correction are applied to capture motion cues.
   
4. **Dataset Generation**  
   - Wide and narrow field-of-view inputs (emulating driver gaze).
   - Balanced class distributions using SMOTE and hybrid sampling.
   
5. **Deep Learning Classification**  
   - Utilises (2+1)D CNN to model spatio-temporal features.
   - Compares performance between RGB and optical flow inputs in wide vs. narrow views.


![Pipeline](https://github.com/user-attachments/assets/1b7ceb47-65ea-462a-a9a2-adf3633d438b)

---
