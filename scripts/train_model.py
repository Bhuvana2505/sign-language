import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ---------------------
# CONFIG
# ---------------------
KEYPOINT_DIR = "keypoints/train"
EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/sign_model.pth"

# ---------------------
# DATASET LOADER
# ---------------------
class SignDataset(Dataset):
    def __init__(self, base_dir):
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(base_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for label in self.classes:
            label_path = os.path.join(base_dir, label)
            for file in os.listdir(label_path):
                if file.endswith(".npy"):
                    self.samples.append(os.path.join(label_path, file))
                    self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = np.load(self.samples[idx], allow_pickle=True)
        # Pad or crop to fixed length
        max_len = 40
        if len(seq) < max_len:
            pad = np.zeros((max_len - len(seq), seq.shape[1]))
            seq = np.vstack((seq, pad))
        else:
            seq = seq[:max_len, :]
        seq = torch.tensor(seq, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label


# ---------------------
# MODEL DEFINITION
# ---------------------
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


# ---------------------
# TRAINING FUNCTION
# ---------------------
def train_model(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    dataset = SignDataset(KEYPOINT_DIR)
    num_classes = len(dataset.classes)
    print(f"Loaded {len(dataset)} samples from {num_classes} classes")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Input size = 75 landmarks × 3 = 225 * 3 = 675
    model = SignLSTM(input_size=225, hidden_size=128, num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 Training started...")
    for epoch in range(EPOCHS):
        loss = train_model(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "class_names": dataset.classes
    }, MODEL_PATH)

    print(f"✅ Training complete. Model saved at {MODEL_PATH}")
