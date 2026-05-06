from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import joblib

app = Flask(__name__)

# LOAD SCALER
scaler = joblib.load("scaler.pkl")

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

# LOAD MODEL
model = LoginDetector()

model.load_state_dict(
    torch.load("model.pth")
)

model.eval()

# HOME PAGE
@app.route("/")

def home():

    return render_template("index.html")

# PREDICT
@app.route("/predict", methods=["POST"])

def predict():

    failed_logins = float(
        request.form["failed_logins"]
    )

    ip_risk = float(
        request.form["ip_risk_score"]
    )

    unusual_time = float(
        request.form["unusual_time"]
    )

    new_device = float(
        request.form["new_device"]
    )

    data = [[
        failed_logins,
        ip_risk,
        unusual_time,
        new_device
    ]]

    data = scaler.transform(data)

    tensor = torch.FloatTensor(data)

    with torch.no_grad():

        prediction = model(tensor)

        probability = prediction.item()

    result = (
        "⚠️ Suspicious Login"
        if probability > 0.5
        else "✅ Normal Login"
    )

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability, 4)
    )

if __name__ == "__main__":

   import os

port = int(os.environ.get("PORT", 5000))

app.run(host="0.0.0.0", port=port)