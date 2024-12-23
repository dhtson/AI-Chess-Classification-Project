import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

def predict(image_path, model, device, transform, classes):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

def load_images_and_predict(folder_path, model_path="chess_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 6
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    classes = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

    image_paths = []
    predictions = []

    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            predicted_label = predict(image_path, model, device, transform, classes)
            image_paths.append(image_path)
            predictions.append(predicted_label)

    return image_paths, predictions

def plot_sample_images(image_paths, predictions):
    plt.figure(figsize=(12, 12))
    num_images = len(image_paths)

    for i in range(min(num_images, 12)):
        plt.subplot(4, 3, i + 1)
        image = Image.open(image_paths[i])
        plt.imshow(image)
        plt.title(f'Label: {predictions[i]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder_path = "test"
    image_paths, predictions = load_images_and_predict(folder_path)
    plot_sample_images(image_paths, predictions)
