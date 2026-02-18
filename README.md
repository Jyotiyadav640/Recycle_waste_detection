♻️ AI-Based Waste Classification for Sustainability

🌍 Project Overview
Waste management is one of the most crucial challenges in achieving environmental sustainability. Improper segregation of waste leads to increased pollution, landfill overflow, and loss of recyclable materials.
This project leverages Artificial Intelligence (AI) and Machine Learning (ML) to automate the process of waste classification into categories such as Organic, Recyclable, and Non-Recyclable waste using Image Classification techniques.
Our goal is to support the Sustainable Development Goals (SDG 11 & 12) — Sustainable Cities and Responsible Consumption — by promoting efficient waste sorting using AI.

🧠 Problem Statement
Manual waste segregation is inefficient, time-consuming, and prone to human error.
We aim to build an AI-powered image classification model that can identify the category of waste items from images, enabling smarter recycling and sustainability practices.

🎯 Objectives

1. Automate waste classification using AI.

2. Reduce human involvement in manual waste segregation.

3. Promote sustainability by improving recycling accuracy.

4.Provide an easy-to-use model that can later be deployed for real-world applications (e.g., smart dustbins, waste monitoring systems).

🔍 Project Features

✅ AI-based image classification using deep learning.
✅ Trained on open-source waste image datasets.
✅ Categorizes waste into multiple classes: Organic, Recyclable, and Non-Recyclable.
✅ Implemented using Python + TensorFlow/Keras.
✅ Includes Jupyter Notebook (.ipynb) for full transparency and reproducibility.
✅ Supports future deployment using Flask or Streamlit for real-time prediction.

🗂️ Dataset

We plan to use an open-source dataset such as:

TrashNet Dataset or Kaggle’s Waste Classification Dataset

Contains images categorized as:

Plastic

Paper

Metal

Glass

Organic

Others (Non-Recyclable)

📦 Download dataset from:

Kaggle: Waste Classification Data

GitHub: TrashNet Dataset

🧩 Tech Stack
1. Programming Language - Python 🐍
2. Libraries - TensorFlow, Keras, NumPy, Pandas, Matplotlib, OpenCV
3. Tools - Jupyter Notebook
4. Dataset Source - Kaggle / TrashNet
5. Deployment (Future Scope) - Streamlit / Flask
6. Version Control - Git + GitHub

⚙️ Project Workflow
Data Collection → Download dataset from Kaggle/TrashNet.

Data Preprocessing → Resize, normalize, and augment images.

Model Building → CNN model using TensorFlow/Keras.

Model Training & Validation → Train and evaluate accuracy & loss metrics.

Prediction & Testing → Classify new images and visualize results.

Deployment (Future Scope) → Deploy model via Streamlit for live demo.

🧰 Setup Instructions

Follow these steps to run this project locally:

1️⃣ Clone the repository
git clone https://github.com/<your-username>/AI-Waste-Classification.git
cd AI-Waste-Classification

## 📦 Model File
The trained CNN model file (`best_model.keras`) is available here:  
👉 [Download from Google Drive](https://drive.google.com/file/d/1ji_6VuLHU6gMPPe-4cka39RsVFi2BPM0/view?usp=drive_link)


2️⃣ Create a virtual environment (recommended)
python -m venv env
source env/bin/activate   # For Mac/Linux
env\Scripts\activate      # For Windows

3️⃣ Install dependencies
pip install -r requirements.txt


If requirements.txt is not provided, manually install:

pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn

4️⃣ Open Jupyter Notebook
jupyter notebook


Then open the file:
waste_classification.ipynb

5️⃣ Run All Cells

Run all cells in order to:

Train the model

Evaluate results

View predictions on test images

📈 Expected Results

Model accuracy of 80–90% on test data.

Real-time image classification capability.

Contribution toward sustainability goals through smart waste management.

🌿 Sustainability Impact

This project supports:

SDG 11 – Sustainable Cities and Communities

SDG 12 – Responsible Consumption and Production

By automating waste segregation, the project encourages:

Cleaner environments 🌱

Better recycling rates ♻️

Reduced landfill waste 🗑️

📘 Folder Structure
📁 AI-Waste-Classification/
│
├── 📄 README.md
├── 📓 waste_classification.ipynb
├── 📁 dataset/
│   ├── train/
│   ├── test/
│   └── validation/
├── 📁 model/
│   └── waste_classifier.h5
└── 📁 images/
    └── sample_predictions/

🏁 Future Scope

1. Improve classification accuracy using transfer learning (ResNet, MobileNet).

2. Develop a mobile/web app for real-time waste recognition.

3. Integrate with IoT-based smart bins for automatic waste sorting.

💚 “Let’s build a cleaner, smarter, and sustainable future with AI.”
