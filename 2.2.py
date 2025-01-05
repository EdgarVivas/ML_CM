import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

from matplotlib.dates import DateFormatter

# Set up the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and Clean Data
data_path = 'aved_raw.csv'  
data = pd.read_csv(
    data_path,
    index_col='time',
    dtype={'time': str}
)

# Preprocess timestamps
data.index = data.index.str.strip()
data.index = pd.to_datetime(data.index, utc=True, errors='coerce')
data = data[~data.index.isna()]

# Feature Selection
features = [
    'BIOLOGY.LINE 3 TANK 1.N2O value',
    'BIOLOGY.LINE 3 TANK 1.NH4 value',
    'BIOLOGY.LINE 3 TANK 1.NO3 value',
    'BIOLOGY.LINE 3 TANK 1.O2 value',
    'BIOLOGY.LINE 3 TANK 1.TEMPERATURE value'
]
data = data[features]
data = data.dropna()

# Define Test Intervals
test_weeks = [
    '2023-01-21 00:00:00+02:00',
    '2023-05-10 00:00:00+02:00',
    '2023-07-10 00:00:00+02:00',
    '2023-11-12 00:00:00+02:00'
]

# Split Data for Training and Testing
test_data_segments = []
train_data = data.copy()

for week in test_weeks:
    start_date = pd.to_datetime(week, utc=True)
    end_date = start_date + pd.Timedelta(days=7)
    weekly_data = train_data.loc[start_date:end_date]
    test_data_segments.append((start_date, end_date, weekly_data))
    train_data = train_data.drop(weekly_data.index)

# Normalize the Training Data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)

# Calculate Threshold for Anomaly Detection
threshold = train_data['BIOLOGY.LINE 3 TANK 1.N2O value'].quantile(0.95)
train_labels = (train_data['BIOLOGY.LINE 3 TANK 1.N2O value'] > threshold).astype(int)

# Create Sequences for Model Input
def generate_sequences(data_array, labels_series, seq_length):
    """
    Creates overlapping sequences of input data and corresponding labels.
    """
    inputs = []
    outputs = []
    for i in range(len(data_array) - seq_length):
        inputs.append(data_array[i:i + seq_length])
        outputs.append(labels_series.iloc[i + seq_length])
    return np.array(inputs), np.array(outputs)

window_size = 10
X_train, y_train = generate_sequences(
    pd.DataFrame(train_scaled, columns=features),
    train_labels,
    window_size
)

# PyTorch Dataset & DataLoader
class AnomalyDetectionDataset(Dataset):
    """
    Custom Dataset for sequence-based anomaly detection.
    """
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AnomalyDetectionDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the LSTM Model
class LSTMAnomalyDetector(nn.Module):
    """
    LSTM-based model for binary classification of anomalies.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMAnomalyDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out.squeeze(dim=1)

input_size = len(features)
hidden_size = 64
num_layers = 1
learning_rate = 0.001
num_epochs = 5

model = LSTMAnomalyDetector(input_size, hidden_size, num_layers).to(device)

# Define Loss Function and Optimizer
criterion = nn.BCELoss()

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
class_weights_tensor = torch.tensor(
    [class_weights_dict[0], class_weights_dict[1]], 
    dtype=torch.float32
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels_batch in train_loader:
        sequences = sequences.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(sequences)
        weights = labels_batch * class_weights_tensor[1] + (1 - labels_batch) * class_weights_tensor[0]
        loss = (criterion(outputs, labels_batch) * weights).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * sequences.size(0)

        preds = (outputs > 0.5).float()
        correct += (preds == labels_batch).sum().item()
        total += labels_batch.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Visualize Predictions by Week
output_dir = "weekly_plots"
os.makedirs(output_dir, exist_ok=True)

def save_weekly_plot(start_date, end_date, week_data, week_index):
    """
    Save a weekly plot showing raw N2O values, predicted probabilities, and actual anomalies.
    """
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        week_data.index,
        week_data['BIOLOGY.LINE 3 TANK 1.N2O value'],
        label='Raw N2O',
        color='red',
        linewidth=2
    )
    ax.set_ylabel("N2O", color='blue')
    ax.tick_params(axis='y', labelcolor='red')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.set_title(f"Week {week_index+1}: {start_date} to {end_date}")

    if len(week_data) < window_size:
        ax.text(
            0.5, 0.5, f'Insufficient data (< {window_size} points)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='red'
        )
        plt.savefig(os.path.join(output_dir, f"week_{week_index+1}.png"), dpi=300)
        plt.close()
        return

    week_scaled = scaler.transform(week_data)
    week_labels = (week_data['BIOLOGY.LINE 3 TANK 1.N2O value'] > threshold).astype(int)

    X_week, y_week = generate_sequences(
        pd.DataFrame(week_scaled, columns=features),
        week_labels,
        window_size
    )

    model.eval()
    X_week_torch = torch.tensor(X_week, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_week_torch)
        preds_prob = outputs.cpu().numpy()
        preds = (outputs > 0.5).float().cpu().numpy()

    time_index = week_data.index[window_size:]

    ax2 = ax.twinx()

    ax2.plot(
        time_index,
        preds_prob,
        label='Predicted Probability',
        color='blue',
        linestyle='--',
        alpha=0.8
    )

    ax2.plot(
        time_index,
        y_week,
        label='Actual Exceedance',
        color='green',
        alpha=0.7
    )

    exceed_pred_indices = np.where(preds == 1)[0]
    if len(exceed_pred_indices) > 0:
        ax2.scatter(
            time_index[exceed_pred_indices],
            preds_prob[exceed_pred_indices],
            color='purple',
            marker='x',
            s=50,
            label='Anomaly'
        )

    ax2.set_ylabel("Probability / Exceedance Label", color='black')
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    file_path = os.path.join(output_dir, f"week_{week_index+1}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()

for i, (start_date, end_date, week_data) in enumerate(test_data_segments):
    save_weekly_plot(start_date, end_date, week_data, i)

print(f"Plots saved in the directory: {output_dir}")
