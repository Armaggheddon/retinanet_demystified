from pathlib import Path
from retinanet import RetinaNet
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path(__file__).parent / "model_detections.jpg"

model = RetinaNet(num_classes=1, backbone_name="resnet18", pretrained_backbone=True, nms_iou_threshold=0.5)

model.load_state_dict(torch.load("retinanet_raccoon_rn18.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

image = Image.open("raccoon_dataset/images/raccoon-1.jpg").convert("RGB")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
preproc_image = transform(image).unsqueeze(0).to(DEVICE)

# Example inference
with torch.no_grad():
    out = model(preproc_image)

overlay = image.copy().resize((224, 224))
draw = ImageDraw.Draw(overlay)

for box, score, label in zip(out[0]['boxes'], out[0]['scores'], out[0]['labels']):
    
    if score < 0.4:
        continue

    box = box.cpu().numpy().astype(int)

    print(f"Label: {label}, Score: {score:.3f}, Box: {box}")


    draw.rectangle(box.tolist(), outline="red", width=2)
    draw.text((box[0], box[1]), f"{label.item()}:{score.item():.2f}", fill="red")   

overlay.save(OUTPUT_DIR)
