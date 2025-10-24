%%writefile app2.py
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import os

# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.load_state_dict(torch.load("fine_tuned_clip.pth", map_location=device))
model.to(device)
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Classes (update this list with the actual class names from your dataset)
classes = [
    'clapping',
    'meeting_and_splitting',
    'sitting',
    'standing_still',
    'walking',
    'walking_while_reading_a_book',
    'walking_while_using_phone'
]

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = np.array(frame) / 255.0
    return (frame * 255).astype(np.uint8)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_features(frames):
    features = []
    for frame in frames:
        frame = preprocess_frame(frame)
        frame = Image.fromarray(frame)
        inputs = processor(images=frame, return_tensors="pt")
        features.append(inputs["pixel_values"].squeeze(0).numpy())
    return np.mean(features, axis=0)  # Average the features

def predict(video_path):
    frames = extract_frames(video_path)
    features = extract_features(frames)
    inputs = torch.tensor(features).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=inputs)
        preds = torch.argmax(outputs, dim=1)
        return classes[preds.item()]

# Streamlit App
st.title("Human Activity Recognition")

# Video upload
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(temp_video_path)

    # Predict the activity in the video
    if st.button("Predict"):
        st.write("Predicting...")
        prediction = predict(temp_video_path)
        st.success(f"Predicted Activity: {prediction}")

    # Clean up the temporary video file after prediction
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)