import torch
import torch.nn as nn

class VisualConfidenceModel(nn.Module):
    """
    Neural Network for classifying confidence based on MediaPipe 
    face/hand landmarks and blendshapes.
    
    Input shape: (Batch, Sequence_Length, Input_Dim)
    Input_Dim is typically 52 (blendshapes) + 126 (hands) + others.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2):
        super(VisualConfidenceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (Batch, Seq, Dim)
        # LSTM output: (Batch, Seq, Hidden * 2)
        lstm_out, _ = self.lstm(x)
        
        # We take the last time step's output for classification
        last_out = lstm_out[:, -1, :]
        
        logits = self.fc(last_out)
        return logits

class AudioConfidenceModel(nn.Module):
    """
    Neural Network for classifying confidence based on Whisper embeddings.
    
    Input shape: (Batch, Sequence_Length, Embedding_Dim)
    Embedding_Dim: 384 for Whisper 'tiny'
    """
    def __init__(self, embedding_dim=384, hidden_dim=128, num_layers=2, num_classes=2):
        super(AudioConfidenceModel, self).__init__()
        # Since Whisper embeddings are already rich, a GRU or LSTM works well
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (Batch, Seq, Embedding_Dim)
        gru_out, _ = self.gru(x)
        
        # Pooling: Mean over time might be better for audio embeddings 
        # to capture the overall sentiment of the 1s window
        pooled = torch.mean(gru_out, dim=1)
        
        logits = self.fc(pooled)
        return logits
