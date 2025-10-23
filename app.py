import torch
import torchvision.transforms as transforms
from models import NeuralNet   
import streamlit as st
from PIL import Image

model = NeuralNet()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device('cpu')))
model.eval()

st.title(" MNIST Digit Classifier")
uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("L")  # grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    img_t = transform(image).unsqueeze(0)  #  batch dimension

    output = model(img_t)
    _, pred = torch.max(output, 1)

    st.image(image, caption=f"Prediction: {pred.item()}", width=150)