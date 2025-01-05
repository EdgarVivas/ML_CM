import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# -------------------------------
# Utility Functions
# -------------------------------
def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# -------------------------------
# SimpleResNet Architecture
# -------------------------------
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleResNet, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Input channels set to 1 for grayscale
        self.bn1 = nn.BatchNorm2d(64)

        # First residual block (no downsampling)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Second residual block (downsampling)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv1x1_1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        # Third residual block (downsampling)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv1x1_2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)

        # Fully connected layer
        # Assuming input images are 64x64, after two downsampling steps: 16x16
        self.fc = nn.Linear(256 * 16 * 16, num_classes)

    def forward(self, x):
        # Initial convolution + batch norm + relu
        x = torch.relu(self.bn1(self.conv1(x)))

        # First residual block
        identity = x
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        x = torch.relu(out + identity)

        # Second residual block (downsampling)
        identity = self.conv1x1_1(x)
        out = torch.relu(self.bn4(self.conv4(x)))
        out = self.bn5(self.conv5(out))
        x = torch.relu(out + identity)

        # Third residual block (downsampling)
        identity = self.conv1x1_2(x)
        out = torch.relu(self.bn6(self.conv6(x)))
        out = self.bn7(self.conv7(out))
        x = torch.relu(out + identity)

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        # Debugging: Print the shape
        expected_features = 256 * 16 * 16
        if x.size(1) != expected_features:
            print(f"Warning: Expected input features to FC layer: {expected_features}, but got {x.size(1)}")
        x = self.fc(x)
        return x

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5, use_amp=False):
    model.train()
    if use_amp:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Debugging: Check device assignment
            if batch_idx == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Inputs on {inputs.device}, Labels on {labels.device}")
                print(f"Model parameters on {next(model.parameters()).device}")

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy for the batch
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    return

# -------------------------------
# Testing Function
# -------------------------------
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Debugging: Check device assignment
            if batch_idx == 0:
                print(f"Test Batch {batch_idx}: Inputs on {inputs.device}, Labels on {labels.device}")
                print(f"Model parameters on {next(model.parameters()).device}")

            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Test Classification Accuracy: {accuracy:.2f}%")
    return accuracy

# -------------------------------
# Experiment Runner
# -------------------------------
def run_experiments(
    data_dir="dataset_bw",
    experiment_params=[
        {'batch_size':32, 'lr':0.01},
        {'batch_size':32, 'lr':0.005},
        {'batch_size':32, 'lr':0.001},
        {'batch_size':32, 'lr':0.0005},
        {'batch_size':16, 'lr':0.01},
        {'batch_size':16, 'lr':0.005},
        {'batch_size':16, 'lr':0.001},
        {'batch_size':16, 'lr':0.0005},
        {'batch_size':64, 'lr':0.01},
        {'batch_size':64, 'lr':0.005},
        {'batch_size':64, 'lr':0.001},
        {'batch_size':64, 'lr':0.0005},
    ],
    num_epochs=5,
    device=None,
    results_dir="results"
):
    """
    Runs multiple training experiments with different hyperparameters.

    Args:
        data_dir (str): Path to the dataset directory.
        experiment_params (list): List of dictionaries with 'batch_size' and 'lr'.
        num_epochs (int): Number of training epochs per experiment.
        device (torch.device): Device to run the experiments on.
        results_dir (str): Directory to save results.

    Returns:
        list: List of dictionaries containing experiment results.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensure_dir(results_dir)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),                         # Convert PIL images to tensors
        # Removed Resize and Normalize for speed
    ])

    # Load the entire dataset using ImageFolder
    full_dataset = ImageFolder(root=data_dir, transform=transform)

    # Get class names
    class_names = full_dataset.classes  # e.g., ['circle_damage', 'line_damage', 'star_damage', 'no_damage', 'multiple_damages']
    num_classes = len(class_names)
    print(f'Classes: {class_names}')

    # Split the dataset into training and testing subsets (80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    results = []

    # Determine if AMP can be used
    use_amp = device.type == 'cuda'

    # Run each experiment
    for idx, params in enumerate(experiment_params):
        experiment_id = idx + 1
        batch_size = params['batch_size']
        lr = params['lr']

        print(f"\n=== Experiment {experiment_id} ===")
        print(f"Batch Size: {batch_size}, Learning Rate: {lr}")

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows to avoid issues
            pin_memory=True if use_amp else False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for Windows to avoid issues
            pin_memory=True if use_amp else False
        )

        # Initialize the model, criterion, and optimizer
        model = SimpleResNet(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train the model
        train_model(model, train_loader, criterion, optimizer, device, num_epochs, use_amp)

        # Test the model
        test_accuracy = test_model(model, test_loader, device)

        # Save the results
        results.append({
            'experiment_id': experiment_id,
            'batch_size': batch_size,
            'lr': lr,
            'accuracy': test_accuracy
        })

    # Save results to CSV
    save_results_to_csv(results, filename=os.path.join(results_dir, "experiment_results.csv"))

    # Print experiment summary
    print("\n===== Experiment Summary =====")
    for r in results:
        print(f"Exp {r['experiment_id']}: Batch Size={r['batch_size']}, LR={r['lr']}, Accuracy={r['accuracy']:.2f}%")

    return results, class_names

# -------------------------------
# Save Results to CSV
# -------------------------------
def save_results_to_csv(results, filename="experiment_results.csv"):
    fieldnames = ['experiment_id', 'batch_size', 'lr', 'accuracy']
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nResults saved to {filename}")

# -------------------------------
# Main Execution
# -------------------------------
def main():
    # Path to the dataset directory containing class subdirectories
    data_dir = 'dataset_bw'  # Update this path to your dataset directory

    # Check if the dataset directory exists
    if not os.path.isdir(data_dir):
        print(f"Dataset directory '{data_dir}' not found. Please ensure the path is correct.")
        return

    # Define experiment parameters (batch sizes and learning rates)
    experiment_params = [
        {'batch_size':32, 'lr':0.01},
        {'batch_size':32, 'lr':0.005},
        {'batch_size':32, 'lr':0.001},
        {'batch_size':32, 'lr':0.0005},
        {'batch_size':16, 'lr':0.01},
        {'batch_size':16, 'lr':0.005},
        {'batch_size':16, 'lr':0.001},
        {'batch_size':16, 'lr':0.0005},
        {'batch_size':64, 'lr':0.01},
        {'batch_size':64, 'lr':0.005},
        {'batch_size':64, 'lr':0.001},
        {'batch_size':64, 'lr':0.0005},
    ]

    # Run experiments
    results, class_names = run_experiments(
        data_dir=data_dir,
        experiment_params=experiment_params,
        num_epochs=5,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        results_dir="results"
    )

if __name__ == "__main__":
    main()
