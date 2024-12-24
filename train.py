# Import required libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix using seaborn heatmap.
    
    Args:
    cm (array): Confusion matrix.
    class_names (list): List of class names for labels.
    """
    plt.figure(figsize=(10, 7))  # Set plot size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Function to train the model
def train_model(data_path, epochs=100, batch_size=64, learning_rate=0.001, model_save_path="chess_model.pth", patience=15):
    """
    Trains a ResNet18 model on the given dataset.

    Args:
    data_path (str): Path to the dataset directory.
    epochs (int): Number of training epochs.
    batch_size (int): Size of each training batch.
    learning_rate (float): Learning rate for the optimizer.
    model_save_path (str): Path to save the best model.
    patience (int): Number of epochs to wait for improvement before early stopping.
    """
    # Check if GPU is available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transformations for preprocessing and augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (input size for ResNet18)
        transforms.RandomRotation(15),  # Randomly rotate images by up to 15 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust color properties
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
    ])

    # Load dataset from the specified directory
    dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
    num_classes = len(dataset.classes)  # Get the number of classes

    # Print class names and their indices
    print(f"Classes: {dataset.classes}")
    print(f"Class-to-Index Mapping: {dataset.class_to_idx}")

    # Split dataset into training and validation sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader objects for batch loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Replace the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)  # Move model to the appropriate device (GPU/CPU)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Learning rate scheduler to reduce learning rate when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Variables to track the best validation loss and early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Lists to store accuracies for plotting
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training step
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)  # Get predicted labels
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        # Compute training accuracy
        train_acc = correct_train / total_train
        train_accuracies.append(train_acc)

        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        # Validation step
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                # Collect predictions and true labels for evaluation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        val_accuracies.append(val_acc)

        # Print metrics for the current epoch
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_acc * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%")

        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)  # Save model
            print(f"Model saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        # Stop training if validation loss does not improve for 'patience' epochs
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Compute and display confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, dataset.classes)

    # Display classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    # Plot training and validation accuracies over epochs
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

    print("Training complete!")

# Main script to initiate the training process
if __name__ == "__main__":
    # Download dataset using kagglehub and train the model
    path = kagglehub.dataset_download('koinguyn/chess-detection')
    data_path = f"{path}"
    train_model(data_path)
