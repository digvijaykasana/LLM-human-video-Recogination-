%%writefile app.py
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Function to extract frames
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

# Preprocess frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = np.array(frame) / 255.0
    return (frame * 255).astype(np.uint8)

# Extract features from frames
def extract_features(frames, processor, device):
    features = []
    for frame in frames:
        frame = preprocess_frame(frame)
        frame = Image.fromarray(frame)
        inputs = processor(images=frame, return_tensors="pt")
        features.append(inputs["pixel_values"].squeeze(0).numpy())
    return np.mean(features, axis=0)  # Average the features

# Predict action
def predict_action(model, video_path, processor, classes, device):
    model.eval()
    frames = extract_frames(video_path)
    features = extract_features(frames, processor, device)
    inputs = torch.tensor([features]).to(device)
    with torch.no_grad():
        video_features = model.get_image_features(pixel_values=inputs)

    text_features = extract_text_features(classes, processor, model, device)
    similarities = cosine_similarity(video_features.cpu().numpy(), text_features)
    max_similarity = np.max(similarities)
    
    predicted_action_index = np.argmax(similarities)
    return classes[predicted_action_index]

# Extract text features
def extract_text_features(texts, processor, model, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.cpu().numpy()

# Streamlit app
def main():
    st.title("Video Action Recognition")
    st.write("Upload a video to get the predicted action.")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save the uploaded file to disk
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(video_path)

        # Initialize model and processor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.load_state_dict(torch.load("fine_tuned_clip_best.pth"))
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Predict action
        actions = ["clapping", "meeting_and_splitting", "sitting", "standing_still", "walking", "walking_while_reading_a_book", "walking_while_using_phone"]
        predicted_action = predict_action(model, video_path, processor, actions, device)

        st.write(f"Predicted Action: *{predicted_action}*")

if __name__ == "__main__":
    main()