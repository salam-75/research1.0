import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================================
# 1. Custom Focal Loss Definition
# =====================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'): 
        # Gamma tuned to 1.5 for better minority class balance
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =====================================================================
# 2. Data Loading & Pre-processing
# =====================================================================
print("1/5: Loading dataset...")
try:
    df = pd.read_csv('mitbih_train.csv', header=None)
except FileNotFoundError:
    print("Default path missing, trying alternative paths...")
    df = pd.read_csv('/content/mitbih_train.csv', header=None)

# Drop any accidental missing rows to prevent runtime errors
df = df.dropna()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.astype(int)

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape input features for 1D-CNN: [Batch, Channel (1), Length (187)]
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Construct PyTorch DataLoaders
train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =====================================================================
# 3. 1D-CNN Architecture Design
# =====================================================================
class ECG1DCNN(nn.Module):
    def __init__(self):
        super(ECG1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 46, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 5) # 5 distinct heartbeat target classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten features
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# =====================================================================
# 4. Environment, Model & Focal Loss Initialization
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECG1DCNN().to(device)

# Tuned class weights to scale precision & recall stability
# Ordering mapping: [Normal, SVPB, PVC, Fusion, Unknown]
weights = torch.tensor([0.5, 1.5, 1.0, 2.5, 1.0], dtype=torch.float).to(device)
criterion = FocalLoss(alpha=weights, gamma=1.5)

optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"2/5: Execution environment ready. Target Device: {device}")

# =====================================================================
# 5. Model Training Loop (5 Epochs)
# =====================================================================
print("\n3/5: Starting deep learning model training with Tuned Focal Loss...")
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"   -> Epoch {epoch+1}/{epochs} Completed. Training Loss: {epoch_loss:.4f}")

# =====================================================================
# 6. Test Evaluation & Visualization
# =====================================================================
print("\n4/5: Initiating model evaluation on testing sequence...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n5/5: Generating target analysis reporting metrics...\n")
# Print Classification Performance Matrices
print("--- Tuned 1D-CNN (Focal Loss) Final Test Report ---")
print(classification_report(all_labels, all_preds, digits=4))

# Confusion Matrix Graphical Plotting Configuration
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Normal', 'SVPB', 'PVC', 'Fusion', 'Unknown'],
            yticklabels=['Normal', 'SVPB', 'PVC', 'Fusion', 'Unknown'])
plt.title('Confusion Matrix - Tuned 1D-CNN with Balanced Focal Loss')
plt.ylabel('True Class Labels')
plt.xlabel('Predicted Class Labels')
plt.show()

