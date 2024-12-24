# Import required libraries
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

# Function to predict the class of a single image
def predict(image_path, model, device, transform, classes):
    """
    Predicts the class of a given image.

    Args:
    image_path (str): Path to the image file.
    model (torch.nn.Module): Pre-trained model used for prediction.
    device (torch.device): Device (CPU/GPU) to perform computations on.
    transform (torchvision.transforms.Compose): Transformations to preprocess the image.
    classes (list): List of class names.

    Returns:
    str: Predicted class label.
    """
    # Open the image and convert it to RGB format
    image = Image.open(image_path).convert("RGB")
    # Apply transformations and add batch dimension
    image = transform(image).unsqueeze(0).to(device)
    # Perform inference
    outputs = model(image)
    # Get the index of the highest score
    _, predicted = torch.max(outputs, 1)
    # Return the class name corresponding to the predicted index
    return classes[predicted.item()]

# Function to load images from a folder and make predictions
def load_images_and_predict(folder_path, model_path="chess_model.pth"):
    """
    Loads images from a folder, predicts their classes using a pre-trained model,
    and returns image paths and their predictions.

    Args:
    folder_path (str): Path to the folder containing images.
    model_path (str): Path to the pre-trained model file.

    Returns:
    list, list: List of image paths and their predicted labels.
    """
    # Select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of classes for the chess pieces
    num_classes = 6
    # Load the pre-trained ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Modify the fully connected layer to match the number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    # Move model to the selected device
    model = model.to(device)
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Define transformations for preprocessing images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (input size for ResNet18)
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
    ])

    # List of class names for the chess pieces
    classes = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

    # Lists to store image paths and their predictions
    image_paths = []
    predictions = []

    # Iterate through files in the specified folder
    for image_name in os.listdir(folder_path):
        # Check if the file is an image (based on extension)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct the full path to the image
            image_path = os.path.join(folder_path, image_name)
            # Predict the label for the image
            predicted_label = predict(image_path, model, device, transform, classes)
            # Store the image path and predicted label
            image_paths.append(image_path)
            predictions.append(predicted_label)

    return image_paths, predictions

# Function to plot a sample of images with their predicted labels
def plot_sample_images(image_paths, predictions):
    """
    Plots a grid of sample images with their predicted labels.

    Args:
    image_paths (list): List of image paths.
    predictions (list): List of predicted labels for the images.
    """
    plt.figure(figsize=(12, 12))  # Set the figure size
    num_images = len(image_paths)  # Get the total number of images

    # Plot up to 12 images in a grid (4x3 layout)
    for i in range(min(num_images, 12)):
        plt.subplot(4, 3, i + 1)  # Define subplot position
        image = Image.open(image_paths[i])  # Open the image
        plt.imshow(image)  # Display the image
        plt.title(f'Label: {predictions[i]}')  # Set the title with the predicted label
        plt.axis('off')  # Remove axes for better visualization

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

# Main script to load images and make predictions
if __name__ == "__main__":
    # Folder containing test images
    folder_path = "test"
    # Load images from the folder and predict their labels
    image_paths, predictions = load_images_and_predict(folder_path)
    # Plot the sample images with predictions
    plot_sample_images(image_paths, predictions)
