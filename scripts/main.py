import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os

# --- 1. Параметры ---
DATA_DIR = r'C:\Users\hayk\Downloads\ai_proj\data'
MODEL_SAVE_DIR = r'C:\Users\hayk\Downloads\ai_proj\models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 4
EPOCHS = 100
NUM_CLASSES = 4
LEARNING_RATE = 0.0001

EARLY_STOP_DELTA = 0.0001    # если loss меняется меньше этого → стоп
EARLY_STOP_PATIENCE = 25     # сколько эпох терпеть

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")
# --- 2. DataLoader (С ОБНОВЛЕННЫМИ АУГМЕНТАЦИЯМИ) ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),

        # --- Геометрические аугментации ---
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),

        # --- Аугментации цвета и стиля (Ваш текущий + новые) ---

        # Случайно изменяем яркость, контраст, насыщенность и оттенок
        # Ваши текущие настройки (0.3, 0.3, 0.3, 0.1) уже достаточно сильные
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),

        # С некоторой вероятностью (p=0.1) превращаем изображение в Ч/Б
        transforms.RandomGrayscale(p=0.1),

        # Добавляем легкое Гауссово размытие
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),

        # --- ДОБАВЛЕННЫЕ (опционально, но полезно) ---

        # "Постеризация": уменьшает количество бит на цветовой канал,
        # что "огрубляет" цвета и заставляет модель не цепляться за мелкие цветовые детали
        transforms.RandomPosterize(bits=4, p=0.1),

        # "Соляризация": инвертирует все значения пикселей выше порога
        # Это очень "неестественная" аугментация, которая ломает привычные паттерны
        transforms.RandomSolarize(threshold=192.0, p=0.1),
        # --- Конец добавленных ---

        transforms.ToTensor(),  # Преобразование в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
    ]),
    'val': transforms.Compose([
        # ... (валидационный сет остается без изменений) ...
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}
#

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# --- 3. Модель ---
class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16, self).__init__()
        self.features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        for param in self.features.parameters():
            param.requires_grad = False
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


model = CustomVGG16(NUM_CLASSES).to(device)

# --- 4. Loss и оптимизатор ---
def categorical_crossentropy(outputs, targets_onehot):
    log_probs = torch.log_softmax(outputs, dim=1)
    loss = -(targets_onehot * log_probs).sum(dim=1).mean()
    return loss

optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# --- 5. Обучение ---
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f"\nЭпоха {epoch+1}/{EPOCHS}")
    print("-" * 20)

    history = {}

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            labels_onehot = torch.nn.functional.one_hot(labels, NUM_CLASSES).float().to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = categorical_crossentropy(outputs, labels_onehot)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

        history[f"{phase}_loss"] = epoch_loss
        history[f"{phase}_acc"] = epoch_acc

        print(f"{phase}: Loss={epoch_loss:.4f} | Acc={epoch_acc:.4f}")

    epoch_time = time.time() - epoch_start
    print(f"⏱ Время эпохи: {epoch_time:.2f} сек")

    # --- Early stopping ---
    if history['val_loss'] + EARLY_STOP_DELTA < best_val_loss:
        print("🔥 Новая лучшая модель (по loss)!")
        best_val_loss = history['val_loss']
        epochs_no_improve = 0

        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))
        print("💾 Сохранено: best_model.pth")

    else:
        epochs_no_improve += 1
        print(f"⚠️ Нет улучшения ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("\n🛑 Early Stopping: обучение остановлено!")
            break

# сохраняем финальную
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "last_epoch.pth"))
print("\n💾 Сохранено: last_epoch.pth")
print("Обучение завершено.")
