# train.py

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn

from model import DigitRecognizer

# Проверка устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Гиперпараметры
batch_size = 64
test_batch_size = 1000
epochs = 50
learning_rate = 0.001

# Преобразования данных
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Загрузка тренировочного и тестового наборов (MNIST)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# Инициализация модели, оптимизатора и функции потерь
model = DigitRecognizer().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = correct / len(test_dataset)
    print(f"Test accuracy: {accuracy:.4f}")

# Обучение модели
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    test()

# Сохранение модели
torch.save(model.state_dict(), "digit_model.pth")
print("Model saved as digit_model.pth")
