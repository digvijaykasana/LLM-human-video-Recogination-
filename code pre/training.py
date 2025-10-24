import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Custom Dataset class
class VideoDataset(Dataset):
    def __init__(self, root_dir, processor, video_extensions=[".mp4", ".avi"]):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.processor = processor
        self.data = []
        self.video_extensions = video_extensions

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            videos = os.listdir(class_dir)
            for video in videos:
                if self.is_video_file(video):
                    video_path = os.path.join(class_dir, video)
                    self.data.append((video_path, class_name))

    def __len__(self):
        return len(self.data)
        
    def is_video_file(self, filename):
        return any(filename.lower().endswith(ext) for ext in self.video_extensions)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self.extract_frames(video_path)
        features = self.extract_features(frames)
        label_idx = self.classes.index(label)
        return features, label_idx

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def extract_features(self, frames):
        features = []
        for frame in frames:
            frame = self.preprocess_frame(frame)
            frame = Image.fromarray(frame)
            inputs = self.processor(images=frame, return_tensors="pt")
            features.append(inputs["pixel_values"].squeeze(0).numpy())
        return np.mean(features, axis=0)  # Average the features

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (224, 224))
        frame = np.array(frame) / 255.0
        return (frame * 255).astype(np.uint8)

# Function to split the dataset into train, validation, and test sets
def split_dataset(dataset, test_size=0.2, val_size=0.2):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_size)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    return train_set, val_set, test_set

# Load Dataset with split
def load_dataset(root_dir, processor, batch_size=4, test_size=0.2, val_size=0.2):
    dataset = VideoDataset(root_dir, processor)
    train_set, val_set, test_set = split_dataset(dataset, test_size, val_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(dataset.classes)

# Fine-tuning the model with validation
def train_model(model, train_loader, val_loader, device, num_classes, num_epochs=10):
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = torch.tensor(inputs).to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.get_image_features(pixel_values=inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

        val_loss = evaluate_model(model, val_loader, device, criterion)
        print(f"Validation Loss after epoch {epoch + 1}: {val_loss:.3f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "fine_tuned_clip_best.pth")

    print('Finished Training')

# Evaluation function for validation
def evaluate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = torch.tensor(inputs).to(device), labels.to(device)
            outputs = model.get_image_features(pixel_values=inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(val_loader)

# Test the model on the test set
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = torch.tensor(inputs).to(device), labels.to(device)
            outputs = model.get_image_features(pixel_values=inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

# Function to extract text features for action classes
def extract_text_features(texts, processor, model, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.cpu().numpy()

# Function to predict action from a video
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
    if max_similarity < 0.2:
        return "Action cannot be detected"
    predicted_action_index = np.argmax(similarities)
    return classes[predicted_action_index]

# Streamlit app for Video Action Recognition
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
    # Training setup
    dataset_dir = "HumanActivityRecognition-VideoDataset"  # Path to the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_loader, val_loader, test_loader, num_classes = load_dataset(dataset_dir, processor)

    # Train the model
    train_model(model, train_loader, val_loader, device, num_classes)
    
    # Test the model
    model.load_state_dict(torch.load("fine_tuned_clip_best.pth"))
    test_model(model, test_loader, device)

    # Run the Streamlit app
    main()