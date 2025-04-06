# visualize.py

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DigitRecognizer

# Проверка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Преобразования для данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Загрузка тестовых данных
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)

# Инициализация модели и загрузка сохранённых весов
model = DigitRecognizer().to(device)
model.load_state_dict(torch.load("digit_model.pth", map_location=device))
model.eval()

# Получение одного батча данных
data_iter = iter(test_loader)
images, labels = next(data_iter)
images = images.to(device)

# Получение предсказаний
with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(dim=1)

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(8, 6))
axes = axes.flatten()
for idx, ax in enumerate(axes):
    ax.imshow(images[idx].cpu().squeeze(), cmap='gray')
    ax.set_title(f"Pred: {preds[idx].item()}\nTrue: {labels[idx].item()}")
    ax.axis('off')
plt.tight_layout()
plt.show()
