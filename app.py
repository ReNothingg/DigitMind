# app.py

import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
from PIL import Image
from model import DigitRecognizer

st.title("Рисуй цифру, а ИИ угадает!")
st.write("Нарисуй цифру в поле ниже:")

# Настройка холста для рисования
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # прозрачный цвет заливки
    stroke_width=10,
    stroke_color="#FFFFFF",         # белый цвет для рисования
    background_color="#000000",     # черный фон
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Преобразование изображения из RGBA в grayscale
    img = canvas_result.image_data.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # Изменение размера до 64x64 (как требуется)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Инвертирование цветов (так как MNIST — белая цифра на черном фоне)
    inverted = 255 - resized

    # Применение пороговой фильтрации для усиления контраста
    _, binarized = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)
    
    # Сохранение debug-изображения для проверки (при желании можно убрать)
    Image.fromarray(binarized).save("debug_digit.png")
    
    # Преобразование изображения в тензор
    img_tensor = transforms.ToTensor()(Image.fromarray(binarized)).unsqueeze(0)
    
    # Вывод отладочной информации
    st.write("Тензор изображения:", img_tensor.shape, img_tensor.min().item(), img_tensor.max().item())
    
    # Загрузка обученной модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitRecognizer().to(device)
    model.load_state_dict(torch.load("digit_model.pth", map_location=device))
    model.eval()
    
    # Предсказание
    with torch.no_grad():
        output = model(img_tensor.to(device))
        pred = output.argmax(dim=1).item()
    
    st.write(f"Предсказанная цифра: **{pred}**")
    st.image(binarized, caption="Обработанное изображение", width=100)
