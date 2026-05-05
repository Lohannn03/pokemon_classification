import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

device = torch.device("cuda")

classes = open("class_names.txt").read().splitlines()

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open("test.jpg")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    _, pred = torch.max(output, 1)

print("Prediction:", classes[pred.item()])