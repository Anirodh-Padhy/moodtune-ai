from tensorflow.keras.models import load_model
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), "../../models/emotion_model.h5")

model = load_model(model_path, compile=False)

emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

def predict_emotion(face):
    face = face / 255.0
    face = np.reshape(face, (1, 128, 128, 3))

    preds = model.predict(face, verbose=0)

    emotion_index = np.argmax(preds)
    emotion = emotion_labels[emotion_index]
    confidence = np.max(preds)

    return emotion, confidence