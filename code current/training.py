import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Custom Dataset class
class VideoDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.classes = sorted(os.listdir(root_dir))
        self.data = []
        
        print("Initializing dataset...")
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            videos = os.listdir(class_dir)
            for video in videos:
                video_path = os.path.join(class_dir, video)
                self.data.append((video_path, class_name))

        print(f"Dataset initialized with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self.extract_frames(video_path)
        features = self.extract_features(frames)
        label_idx = self.classes.index(label)
        return torch.tensor(features).to(device), torch.tensor(label_idx).to(device)  # Move data to GPU

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

# Load and split dataset
def load_and_split_dataset(root_dir, processor, batch_size=4, val_split=0.2, test_split=0.1):
    print("Loading and splitting dataset...")
    dataset = VideoDataset(root_dir, processor)
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split into {train_size} training samples, {val_size} validation samples, and {test_size} test samples.")
    
    return train_loader, val_loader, test_loader, len(dataset.classes)

def train_and_validate(model, train_loader, val_loader, device, num_classes, num_epochs=10):
    print("Starting training and validation...")
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_train_loss = 0.0
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

            optimizer.zero_grad()
            outputs = model.get_image_features(pixel_values=inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_losses.append(running_train_loss / len(train_loader))
        print(f"Training loss: {train_losses[-1]:.4f}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                print(f"Validating batch {batch_idx + 1}/{len(val_loader)}...")
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                outputs = model.get_image_features(pixel_values=inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_losses.append(running_val_loss / len(val_loader))
        val_accuracies.append(accuracy_score(all_labels, all_preds))
        print(f"Validation loss: {val_losses[-1]:.4f}, Validation accuracy: {val_accuracies[-1]:.4f}")

    torch.save(model.state_dict(), "fine_tuned_clip.pth")
    print("Model training complete and saved.")
    return train_losses, val_losses, val_accuracies

def evaluate_and_plot(model, test_loader, device, classes):
    print("Evaluating model on test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            print(f"Evaluating batch {batch_idx + 1}/{len(test_loader)}...")
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model.get_image_features(pixel_values=inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

    return test_accuracy

def plot_training_curves(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    print("Plotting training curves...")
    plt.figure(figsize=(12, 5))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_dir = "HumanActivityRecognition-VideoDataset"  # Path to your dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load and split dataset
    train_loader, val_loader, test_loader, num_classes = load_and_split_dataset(dataset_dir, processor)

    # Train and validate the model
    train_losses, val_losses, val_accuracies = train_and_validate(model, train_loader, val_loader, device, num_classes)

    # Evaluate and plot results
    test_accuracy = evaluate_and_plot(model, test_loader, device, train_loader.dataset.dataset.classes)
    plot_training_curves(train_losses, val_losses, val_accuracies)