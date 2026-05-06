import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# LOAD CSV
df = pd.read_csv("logs.csv")

# INPUTS
X = df[[
    "failed_logins",
    "ip_risk_score",
    "unusual_time",
    "new_device"
]].values

# OUTPUT
y = df[["attack"]].values

# NORMALIZE
scaler = StandardScaler()

X = scaler.fit_transform(X)

# SAVE SCALER
joblib.dump(scaler, "scaler.pkl")

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TENSORS
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# MODEL
class LoginDetector(nn.Module):

    def __init__(self):

        super(LoginDetector, self).__init__()

        self.layer1 = nn.Linear(4, 16)

        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(16, 8)

        self.layer3 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.sigmoid(x)

        return x

model = LoginDetector()

# LOSS
criterion = nn.BCELoss()

# OPTIMIZER
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# TRAIN
epochs = 1000

for epoch in range(epochs):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch+1) % 100 == 0:

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {loss.item():.4f}"
        )

# SAVE MODEL
torch.save(
    model.state_dict(),
    "model.pth"
)

print("\nMODEL TRAINED SUCCESSFULLY")