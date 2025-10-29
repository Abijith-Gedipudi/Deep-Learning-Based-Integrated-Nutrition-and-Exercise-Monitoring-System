# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FoodClassifier(nn.Module):
    """ResNet50-based food classifier"""
    def __init__(self, num_classes=101):
        super(FoodClassifier, self).__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

def preprocess_image(image_pil):
    """Preprocess PIL image for model inference"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image_pil).unsqueeze(0)

def load_model(checkpoint_path='food101_model_for_inference (1).pth', device=None):
    """Load trained model from checkpoint"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract class names
    class_names = checkpoint.get('class_names') or checkpoint.get('classes')
    if class_names is None:
        raise ValueError("Checkpoint must contain 'class_names' or 'classes' key")
    
    num_classes = len(class_names)
    
    # Initialize model
    model = FoodClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, device

def predict_food(model, image_pil, class_names, device, topk=5):
    """
    Predict food class from image
    
    Args:
        model: Trained FoodClassifier model
        image_pil: PIL Image object
        class_names: List of class names
        device: torch device
        topk: Number of top predictions to return
    
    Returns:
        List of tuples (class_name, probability)
    """
    img_tensor = preprocess_image(image_pil).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        topk_prob, topk_idx = torch.topk(probabilities, min(topk, len(class_names)))
    
    results = []
    for prob, idx in zip(topk_prob[0], topk_idx[0]):
        class_name = class_names[idx.item()]
        probability = float(prob.item())
        results.append((class_name, probability))
    
    return results