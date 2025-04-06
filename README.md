# DigitMind

DigitMind 🧠 is a Python-based web app that recognizes handwritten digits in real time. Users can draw a digit on the screen, and an AI model will attempt to guess which number it is. This project uses a fully connected neural network trained on the MNIST dataset, resized to 64x64 for flexibility.

---

## Features

- ✏️ Draw digits directly in the browser
- 🤖 Neural network powered by PyTorch
- ⚖️ Customizable input size (64x64 pixels)
- ✨ Clean Streamlit UI
- ✅ Easy to extend and retrain

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DigitMind.git
   cd DigitMind
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:
   ```bash
   python train.py
   ```

5. Run the app:
   ```bash
   python -m streamlit run app.py
   ```

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Streamlit
- streamlit-drawable-canvas
- OpenCV
- Pillow

---

## File Structure

```
DigitMind/
├── app.py               # Streamlit app
├── train.py             # Training script
├── model.py             # Neural network model definition
├── digit_model.pth      # Saved trained model (after running train.py)
├── requirements.txt     # Dependencies
├── debug_digit.png      # Optional debug image
└── README.md            # You're here :)
```

---

## Author

Created by ReNothing. Contributions and feedback welcome!

