import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils

# --- 1. Define Model Architecture (Must be same as in training) ---
# This is necessary for torch.load() to know how to structure the model

latent_dim = 100
n_classes = 10
img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat((label_emb, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# --- 2. Load the Trained Model ---

# Use a cache to load the model only once
@st.cache_resource
def load_model():
    model = Generator()
    # Load the saved state dictionary. Make sure the file is in the same directory.
    # The map_location argument ensures the model loads correctly whether on CPU or GPU.
    model.load_state_dict(torch.load('cgan_generator.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

generator = load_model()
device = torch.device('cpu') # The app will run on CPU

# --- 3. Streamlit Web App Interface ---

st.set_page_config(layout="wide", page_title="Handwritten Digit Generator")

st.title("✍️ Handwritten Digit Generator")
st.write(
    "This web application uses a Conditional Generative Adversarial Network (cGAN) "
    "trained on the MNIST dataset to generate images of handwritten digits. "
    "Select a digit from the dropdown below and click 'Generate' to see the results."
)

st.sidebar.header("Controls")
# User selects the digit to generate
selected_digit = st.sidebar.selectbox("Choose a digit (0-9)", list(range(10)))

# Button to trigger generation
if st.sidebar.button("Generate Images"):
    st.subheader(f"Generated Images for Digit: {selected_digit}")

    with st.spinner('Generating...'):
        # --- 4. Image Generation Logic ---
        num_images = 5
        
        # Prepare latent vectors (noise)
        noise = torch.randn(num_images, latent_dim, device=device)
        
        # Prepare labels
        labels = torch.LongTensor([selected_digit] * num_images).to(device)

        # Generate images
        with torch.no_grad():
            generated_imgs = generator(noise, labels)

        # Post-process images for display
        # Rescale images from [-1, 1] to [0, 1]
        generated_imgs = 0.5 * generated_imgs + 0.5

        # Use columns to display images side-by-side
        cols = st.columns(num_images)
        for i, col in enumerate(cols):
            with col:
                st.image(
                    generated_imgs[i].cpu().numpy().squeeze(),
                    caption=f"Image {i+1}",
                    width=100,
                    use_column_width='auto' # Adjust image display
                )
else:
    st.info("Select a digit and click the 'Generate Images' button in the sidebar to start.")