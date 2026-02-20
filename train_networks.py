import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import VisualConfidenceModel, AudioConfidenceModel
from dataset import ConfidenceDataset
import os

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, model_name="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training for {model_name} on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        val_acc = 0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_acc = 100 * val_correct / val_total
            
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to {model_name}\n")

def main():
    # 1. VISUAL NETWORK
    print("=== Training Visual Confidence Network ===")
    visual_ds = ConfidenceDataset(mode='visual')
    
    if len(visual_ds) > 0:
        # Split into train/val (simple 80/20)
        train_size = int(0.8 * len(visual_ds))
        val_size = len(visual_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(visual_ds, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        # Input dim: 52 (blendshapes) + 126 (hands) = 178
        visual_model = VisualConfidenceModel(input_dim=178)
        train_model(visual_model, train_loader, val_loader, model_name="visual_confidence.pth")
    else:
        print("No visual data found. Run process_data.py first.")

    # 2. AUDIO NETWORK
    print("=== Training Audio Confidence Network ===")
    audio_ds = ConfidenceDataset(mode='audio')
    
    if len(audio_ds) > 0:
        train_size = int(0.8 * len(audio_ds))
        val_size = len(audio_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(audio_ds, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        # Whisper tiny embedding dim is 384
        audio_model = AudioConfidenceModel(embedding_dim=384)
        train_model(audio_model, train_loader, val_loader, model_name="audio_confidence.pth")
    else:
        print("No audio data found. Run process_data.py first.")

if __name__ == "__main__":
    main()
