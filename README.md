# AI Sign Language Recognizer

## Overview
This project implements an AI-based Sign Language Recognition system that detects hand gestures and converts them into readable text. The system uses computer vision and machine learning techniques to recognize sign language gestures captured through a webcam.

The goal of the project is to improve communication accessibility for individuals with hearing or speech impairments.

---

## Features
- Real-time hand gesture detection
- Sign language gesture classification
- Text output for recognized signs
- Machine learning-based gesture recognition
- Scalable system for adding more gestures

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn / TensorFlow

---

## Project Structure

ai-sign-language-recognizer
│
├── dataset/        # Dataset used for training
├── models/         # Trained machine learning models
├── src/            # Main source code
│   └── main.py
│
├── requirements.txt
├── README.md
└── demo.png

---

## How It Works

1. Capture hand gestures using webcam
2. Preprocess captured frames
3. Extract gesture features
4. Feed features into trained machine learning model
5. Predict the corresponding sign language gesture
6. Convert prediction into readable text

---

## Installation

Clone the repository

git clone https://github.com/adityautkarsh52-art/ai-sign-language-recognizer.git

Go to project directory

cd ai-sign-language-recognizer

Install dependencies

pip install -r requirements.txt

Run the project

python main.py

---

## Future Improvements

- Support for more gestures
- Improved recognition accuracy
- Deploy as a web application
- Add speech output for detected text

---

## Author

Aditya Utkarsh  
B.Tech – Artificial Intelligence & Machine Learning  
Bennett University
