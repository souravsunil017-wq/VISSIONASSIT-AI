"""
Fast Fine-Tuning of Pre-Trained MobileNetV3 Neural Network on Indian Currency Dataset.
Trained with diverse background & negative samples to eliminate false triggers.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image, ImageDraw

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

CLASSES = ["10", "20", "50", "100", "200", "500", "background"]
DATASET_DIR = os.path.join(os.path.dirname(__file__), "currency_dataset")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "indian_currency_mobilenet.pth")

def generate_background_samples():
    bg_dir = os.path.join(DATASET_DIR, "background")
    os.makedirs(bg_dir, exist_ok=True)
    
    # 1. Plain white sheets with handwritten numbers (Anti-spoof)
    for i in range(40):
        img = Image.new("RGB", (300, 200), color=(random.randint(220, 255), random.randint(220, 255), random.randint(220, 255)))
        draw = ImageDraw.Draw(img)
        num_str = random.choice(["500", "200", "100", "50", "20", "10", "Rupees", "Notepad", "Bank", "Note", "10 Rs", "Money"])
        draw.text((random.randint(30, 160), random.randint(30, 120)), num_str, fill=(random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)))
        img.save(os.path.join(bg_dir, f"paper_spoof_{i}.jpg"))
        
    # 2. Desk, wood, wall textures
    for i in range(40):
        base_col = np.array([random.randint(40, 180), random.randint(40, 180), random.randint(40, 180)], dtype=np.uint8)
        noise = np.random.randint(-30, 30, (200, 300, 3), dtype=np.int16)
        arr = np.clip(base_col + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(bg_dir, f"desk_wall_{i}.jpg"))

    # 3. Dark room / shadow backgrounds
    for i in range(40):
        arr = np.random.randint(0, 60, (200, 300, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(bg_dir, f"dark_room_{i}.jpg"))

    # 4. Skin tone / hand simulator
    for i in range(40):
        skin_r = random.randint(140, 230)
        skin_g = int(skin_r * random.uniform(0.65, 0.82))
        skin_b = int(skin_g * random.uniform(0.60, 0.78))
        arr = np.full((200, 300, 3), (skin_r, skin_g, skin_b), dtype=np.uint8)
        noise = np.random.randint(-15, 15, (200, 300, 3), dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(bg_dir, f"skin_hand_{i}.jpg"))

class CurrencyDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, samples_per_class=250, is_train=True, val_split=0.15):
        self.samples = []
        self.transform = transform
        
        for idx, cls_name in enumerate(CLASSES):
            cls_dir = os.path.join(dataset_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.seed(42 + idx)
            random.shuffle(files)
            
            selected = files[:samples_per_class] if samples_per_class and len(files) > samples_per_class else files
            split_idx = int(len(selected) * (1 - val_split))
            
            cls_files = selected[:split_idx] if is_train else selected[split_idx:]
            for f in cls_files:
                self.samples.append((f, idx))
                
        random.shuffle(self.samples)
        phase = "Training" if is_train else "Validation"
        print(f"[Dataset {phase}] Loaded {len(self.samples)} total samples across {len(CLASSES)} classes.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
            
        if self.transform:
            img = self.transform(img)
        return img, label

def train_model():
    print("=" * 65, flush=True)
    print("  TRAINING PRE-TRAINED MOBILENETV3 ON INDIAN CURRENCY DATASET", flush=True)
    print("=" * 65, flush=True)
    
    generate_background_samples()
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CurrencyDataset(DATASET_DIR, transform=train_transform, samples_per_class=250, is_train=True)
    val_dataset = CurrencyDataset(DATASET_DIR, transform=val_transform, samples_per_class=250, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Load Pre-Trained MobileNetV3 Small
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    
    # Unfreeze upper features for fine-grained currency texture/motif learning
    for param in model.features[:3].parameters():
        param.requires_grad = False
        
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(CLASSES))
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.features[3:].parameters(), 'lr': 2e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    epochs = 6
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_loss = total_loss / total
        train_acc = (correct / total) * 100
        
        # Validation pass
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for v_images, v_labels in val_loader:
                v_images, v_labels = v_images.to(device), v_labels.to(device)
                v_out = model(v_images)
                _, v_preds = torch.max(v_out, 1)
                v_correct += (v_preds == v_labels).sum().item()
                v_total += v_labels.size(0)
        val_acc = (v_correct / max(1, v_total)) * 100
        
        print(f"  Epoch [{epoch+1}/{epochs}] -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%", flush=True)
        
        if val_acc > best_val_acc or epoch == epochs - 1:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "arch": "mobilenet_v3_small"
            }, MODEL_SAVE_PATH)
    
    print(f"\n[SUCCESS] Best trained model (Val Acc: {best_val_acc:.1f}%) saved to: {MODEL_SAVE_PATH}", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    train_model()
