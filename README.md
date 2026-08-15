---
title: Herb-AI Clinical RAG & Vision Agent
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# HerbID: AI-Powered Botanical Identifier & Medicinal Guide

An end-to-end Data Science and Full-Stack project utilizing Computer Vision to classify medicinal herbs from images/video, paired with a RAG (Retrieval-Augmented Generation) pipeline to retrieve traditional and scientific clinical usages.

## 🚀 Core Capabilities
- **Botanical Identification:** Classifies herb species using an optimized YOLO/PyTorch inference pipeline.
- **Clinical Context:** Retrieves medicinal properties, biological data, and traditional usages via a structured RAG engine.
- **Full-Stack Integration:** Seamless interaction between a React frontend and a robust FastAPI backend.

## 📂 Repository Structure
- `research/`: Jupyter notebooks exploring data augmentation, model training, and performance metrics (Confusion Matrix, F1-Score).
- `backend/`: FastAPI application serving inference endpoints and managing database connections.
- `frontend/`: React application providing the user interface for image uploads and analysis results.

## 📊 Model Performance
*(Currently under evaluation - Metrics such as Validation Accuracy, Precision, and Recall are tracked via experiments in `/research`)*

## 🛠️ Tech Stack
- **Languages:** Python, JavaScript (React)
- **Frameworks:** PyTorch, FastAPI, Next.js
- **Database:** SQLite/SQL
- **Deployment:** Docker, Hugging Face Spaces

---

### Getting Started
1. **Clone the repository:** 
   `git clone https://github.com/Steven-Ou/MedAi.git`
2. **Install dependencies:** 
   `pip install -r backend/requirements.txt`
3. **Run the application:** 
   `./start.sh`

---
*Developed by Steven Ou | Queens College 2027*