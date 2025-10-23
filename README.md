# 🧠 MNIST Digit Classifier using PyTorch

A deep learning project that classifies handwritten digits (0–9) from the **MNIST dataset** using a trained **Convolutional Neural Network (CNN)** built with **PyTorch**.  
This project also includes a **Streamlit-based web app** that allows users to upload handwritten digit images and get real-time predictions.

---

## 🚀 Features

- 🧩 **Digit Recognition:** Predicts digits (0–9) from handwritten images  
- 💡 **CNN Model:** Built using PyTorch with ReLU activation and Softmax output  
- 🖼️ **User Interface:** Streamlit app for interactive digit uploads and predictions  
- 💾 **Model Persistence:** Trained model saved and loaded via `mnistmodel.pth`  
- 📊 **Accurate Results:** Achieves over 98% accuracy on test data  

---
<img width="968" height="871" alt="image" src="https://github.com/user-attachments/assets/c82128fa-b5b4-4549-9f59-1c35bb7907b9" />

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python |
| **Framework** | PyTorch |
| **Web Interface** | Streamlit |
| **Dataset** | MNIST (handwritten digits) |
| **Libraries** | Torch, Torchvision, PIL, NumPy |

---
```
MNIST-Digit-Classifier/
│
├── model.py # Defines the CNN model architecture
├── train_model.ipynb # Notebook for training and saving the model
├── mnistmodel.pth # Saved PyTorch model
├── app.py # Streamlit app for digit classification
├── requirements.txt # List of dependencies
└── README.md # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anas-fareedi/MNIST-Digit-Classifier.git
   cd MNIST-Digit-Classifier
Create and activate a virtual environment (recommended):
```
python -m venv venv
venv\Scripts\activate        # On Windows
# source venv/bin/activate   # On macOS/Linux
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the Streamlit app:
```
streamlit run app.py
```
## Usage

Upload an image of a handwritten digit (.png, .jpg, etc.)

The app will preprocess the image (convert to grayscale, resize to 28x28)

The trained CNN model will predict the digit and display the result instantly

## Model Overview

Input layer: 1×28×28 grayscale image

Hidden layers: 2 convolutional + ReLU + MaxPooling layers

Fully connected layers: Linear layers for classification

Output layer: 10 neurons (for digits 0–9)

## Results
Metric	Value
Training Accuracy	~99%
Test Accuracy	~98%
Loss Function	CrossEntropyLoss
Optimizer	Adam
🧑‍💻 Author

Anas Fareedi
AI & ML Developer | Python Enthusiast | Backend Developer
🔗 anas-fareedi

## License

This project is licensed under the MIT License — you’re free to use, modify, and distribute it with proper credit.

⭐ If you found this project helpful, don’t forget to give it a star on GitHub!
