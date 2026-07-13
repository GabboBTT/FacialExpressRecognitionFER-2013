import numpy as np
from skimage.io import imread
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Caricamento modello e dati
model = tf.keras.models.load_model('../models/mini_xception_best.h5')
TEST_DIR = '../dataset/test'

def get_brightness(img):
    return np.mean(img)

def run_bias_analysis():
    print("🔍 Analisi del Bias (Luminosità) in corso...")
    
    images, labels, brightnesses = [], [], []
    classes = sorted(os.listdir(TEST_DIR))
    
    for label_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(TEST_DIR, cls_name)
        for file in os.listdir(cls_dir):
            img = imread(os.path.join(cls_dir, file), as_gray=True)
            images.append(img)
            labels.append(label_idx)
            brightnesses.append(get_brightness(img))
    
    images = np.array(images).reshape(-1, 48, 48, 1)
    brightnesses = np.array(brightnesses)
    
    # Dividiamo il dataset per luminosità (Median split)
    median_b = np.median(brightnesses)
    dark_mask = brightnesses < median_b
    bright_mask = brightnesses >= median_b
    
    # Valutiamo le performance
    preds = np.argmax(model.predict(images), axis=1)
    
    acc_dark = accuracy_score(np.array(labels)[dark_mask], preds[dark_mask])
    acc_bright = accuracy_score(np.array(labels)[bright_mask], preds[bright_mask])
    
    print(f"\n📊 RISULTATI ANALISI BIAS:")
    print(f"Accuratezza su immagini SCURE: {acc_dark:.4f}")
    print(f"Accuratezza su immagini CHIARE: {acc_bright:.4f}")
    print(f"Differenziale di Bias: {abs(acc_dark - acc_bright):.4f}")

if __name__ == '__main__':
    run_bias_analysis()