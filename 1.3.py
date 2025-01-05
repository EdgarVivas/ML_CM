


import os
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from collections import defaultdict

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

class OriginalResNet(nn.Module):
    """Original ResNet-like architecture without enhancements."""
    def __init__(self, num_classes=5):
        super(OriginalResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual Block 1
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)

        # Residual Block 2 (Downsampling)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.downsample1 = nn.Conv2d(64, 128, 1, 2, 0)

        # Residual Block 3 (Downsampling)
        self.conv6 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256)
        self.downsample2 = nn.Conv2d(128, 256, 1, 2, 0)

        self.fc = nn.Linear(256 * 16 * 16, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Block 1
        identity = x
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        x = F.relu(out + identity)

        # Residual Block 2
        identity = self.downsample1(x)
        out = F.relu(self.bn4(self.conv4(x)))
        out = self.bn5(self.conv5(out))
        x = F.relu(out + identity)

        # Residual Block 3
        identity = self.downsample2(x)
        out = F.relu(self.bn6(self.conv6(x)))
        out = self.bn7(self.conv7(out))
        x = F.relu(out + identity)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EnhancedResNet(nn.Module):
    """Improved ResNet-like architecture with Dropout and Global Average Pooling."""
    def __init__(self, num_classes=5):
        super(EnhancedResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual Block 1
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)

        # Residual Block 2 (Downsampling)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.downsample1 = nn.Conv2d(64, 128, 1, 2, 0)

        # Residual Block 3 (Downsampling)
        self.conv6 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256)
        self.downsample2 = nn.Conv2d(128, 256, 1, 2, 0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Block 1
        identity = x
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        x = F.relu(out + identity)

        # Residual Block 2
        identity = self.downsample1(x)
        out = F.relu(self.bn4(self.conv4(x)))
        out = self.bn5(self.conv5(out))
        x = F.relu(out + identity)

        # Residual Block 3
        identity = self.downsample2(x)
        out = F.relu(self.bn6(self.conv6(x)))
        out = self.bn7(self.conv7(out))
        x = F.relu(out + identity)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train(model, loader, criterion, optimizer, device, epochs=5, scheduler=None):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if scheduler:
            scheduler.step()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return losses

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def execute_experiments(data_dir, params_list, epochs, device, save_dir):
    create_directory(save_dir)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    num_classes = len(dataset.classes)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    results = []
    for idx, params in enumerate(params_list):
        print(f"\n--- Experiment {idx+1}: {params['model_name']} ---")
        batch = params['batch_size']
        lr = params['lr']
        wd = params['weight_decay']
        sched = params['use_scheduler']
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)
        model = params['model_class'](num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler_step = None
        if sched:
            scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        loss_curve = train(model, train_loader, criterion, optimizer, device, epochs, scheduler_step)
        acc = evaluate(model, test_loader, device)
        results.append({
            'id': idx+1,
            'name': params['model_name'],
            'batch_size': batch,
            'lr': lr,
            'weight_decay': wd,
            'scheduler': sched,
            'final_loss': loss_curve[-1],
            'accuracy': acc,
            'loss_curve': loss_curve
        })
    save_results(results, os.path.join(save_dir, "results.csv"))
    plot_losses(results, save_dir)
    plot_accuracies(results, save_dir)
    summarize(results)

def save_results(results, filepath):
    fields = ['id', 'name', 'batch_size', 'lr', 'weight_decay', 'scheduler', 'final_loss', 'accuracy']
    with open(filepath, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for res in results:
            writer.writerow({key: res[key] for key in fields})
    print(f"Results saved to {filepath}")

def plot_losses(results, save_dir):
    plt.figure()
    for res in results:
        plt.plot(res['loss_curve'], label=res['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "losses.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Loss plot saved to {path}")

def plot_accuracies(results, save_dir):
    names = [res['name'] for res in results]
    acc = [res['accuracy'] for res in results]
    plt.figure()
    plt.bar(names, acc, color=['blue', 'green'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    for i, v in enumerate(acc):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    plt.ylim(0, 100)
    plt.tight_layout()
    path = os.path.join(save_dir, "accuracies.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Accuracy plot saved to {path}")

def summarize(results):
    print("\n=== Summary ===")
    for res in results:
        print(f"Model: {res['name']}, Batch: {res['batch_size']}, LR: {res['lr']}, "
              f"WD: {res['weight_decay']}, Scheduler: {res['scheduler']}, "
              f"Final Loss: {res['final_loss']:.4f}, Accuracy: {res['accuracy']:.2f}%")

def main():
    data_directory = 'dataset_bw'  # Update this path as needed
    if not os.path.isdir(data_directory):
        print(f"Data directory '{data_directory}' not found.")
        return
    experiments = [
        {
            'model_name': 'Original',
            'model_class': OriginalResNet,
            'batch_size': 32,
            'lr': 0.01,
            'weight_decay': 0.0,
            'use_scheduler': False
        },
        {
            'model_name': 'Enhanced',
            'model_class': EnhancedResNet,
            'batch_size': 32,
            'lr': 0.01,
            'weight_decay': 1e-4,
            'use_scheduler': True
        }
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_directory = "results_plots"
    execute_experiments(data_directory, experiments, epochs=10, device=device, save_dir=results_directory)

if __name__ == "__main__":
    main()



