import cv2
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import argparse

DEFAULT_MODEL_PATH = r"C:\Users\hayk\Downloads\ai_proj\models\best_model_angun_agment.pth"
CLASS_NAMES = ['glass', 'metal', 'paper', 'plastic']
IMG_SIZE = (224, 224)
CAMERA_INDEX = 0
USE_GPU = True

class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        for p in self.features.parameters():
            p.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

def load_model(path: str, num_classes: int, device):
    model = CustomVGG16(num_classes=num_classes)
    model.to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded model: {path}")
    return model

preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_frame(model, device, pil_image):
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf, probs

def main(model_path, source, class_names, use_gpu=True):
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print("Using:", device)

    model = load_model(model_path, len(class_names), device)

    is_file = os.path.isfile(source)
    cap = cv2.VideoCapture(source if is_file else int(source))

    if not cap.isOpened():
        print("❌ Cannot open camera:", source)
        return

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)

    BLUR_KERNEL_SIZE = (31, 31)

    fps = 0.0
    prev_time = time.time()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        box_size = int(h * 0.50)
        y1 = (h - box_size) // 2
        y2 = y1 + box_size
        x1 = (w - box_size) // 2
        x2 = x1 + box_size

        crop = frame[y1:y2, x1:x2]

        display_frame = cv2.GaussianBlur(frame, BLUR_KERNEL_SIZE, 0)
        display_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        t0 = time.time()
        pred_idx, conf, probs = predict_frame(model, device, pil)
        t1 = time.time()

        label = f"{class_names[pred_idx]}: {conf * 100:.1f}%"
        inf_time_ms = (t1 - t0) * 1000

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - prev_time))
        prev_time = now

        info = f"FPS: {fps:.1f}  Infer: {inf_time_ms:.1f}ms"

        for i, cname in enumerate(class_names):
            p = probs[i]
            bar_x = 10
            bar_y = display_frame.shape[0] - 120 + i * 25
            bar_w = int(p * 200)
            cv2.putText(display_frame, cname, (bar_x, bar_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)
            cv2.rectangle(display_frame, (bar_x + 60, bar_y - 12),
                          (bar_x + 60 + bar_w, bar_y + 5),
                          (0, 255, 0), -1)

        cv2.putText(display_frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, info, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Garbage Detector", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--source", type=str, default=str(CAMERA_INDEX))
    parser.add_argument("--nogpu", action="store_true")
    args = parser.parse_args()

    main(args.model, args.source, CLASS_NAMES, use_gpu=not args.nogpu)
