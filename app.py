import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------
# Generator Definition
# ---------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, img_dim=784):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_embed(labels)
        x = torch.cat([noise, labels], dim=1)
        return self.model(x)

# ---------------------
# Load Trained Generator
# ---------------------
@st.cache_resource
def load_generator(model_path="generator.pth"):
    device = 'cpu'
    G = Generator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    return G

# ---------------------
# Digit Generation Function
# ---------------------
def generate_digits(generator, digit=0, n=5, noise_dim=100):
    device = 'cpu'
    with torch.no_grad():
        noise = torch.randn(n, noise_dim).to(device)
        labels = torch.full((n,), digit, dtype=torch.long).to(device)
        images = generator(noise, labels)
        images = images.view(-1, 28, 28).cpu().numpy()
        images = (images + 1) / 2  # scale from [-1,1] to [0,1]
    return images

# ---------------------
# Streamlit App
# ---------------------
st.title("🧠 Handwritten Digit Generator")
st.write("Select a digit (0–9) and generate 5 MNIST-style images.")

digit = st.slider("Select digit", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    st.write(f"Generating 5 images of digit `{digit}`...")
    generator = load_generator()
    images = generate_digits(generator, digit=digit)

    cols = st.columns(5)
    for i in range(5):
        img = (images[i] * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        cols[i].image(img_pil, use_container_width=True, caption=f"Digit {digit}")

