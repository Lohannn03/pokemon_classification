import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

# ===== LOAD MODEL =====
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 150)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# ===== TARGET LAYER =====
target_layer = model.layer4[-1]

# ===== HOOK =====
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ===== IMAGE LOAD =====
img_path = "test.jpg"  # 테스트 이미지 넣어라
img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

input_tensor = transform(img).unsqueeze(0)

# ===== FORWARD =====
output = model(input_tensor)
pred_class = output.argmax().item()

# ===== BACKWARD =====
model.zero_grad()
output[0, pred_class].backward()

# ===== GRAD-CAM =====
grad = gradients[0].squeeze().detach().numpy()
feat = features[0].squeeze().detach().numpy()

weights = np.mean(grad, axis=(1,2))

cam = np.zeros(feat.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * feat[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224,224))
cam = cam - cam.min()
cam = cam / cam.max()

# ===== OVERLAY =====
img_np = np.array(img.resize((224,224)))

heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
overlay = heatmap * 0.4 + img_np

# ===== SHOW =====
cv2.imshow("Grad-CAM", overlay.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()