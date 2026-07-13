import os
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

TRAIN_DIR = '../dataset/train'
TEST_DIR = '../dataset/test'

def load_data_and_extract_hog(data_dir):
    print(f"⏳ Estrazione feature HOG da: {data_dir}... (Potrebbe richiedere un paio di minuti)")
    X, y = [], []
    classes = sorted(os.listdir(data_dir))
    
    for label_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir): continue
        
        for file in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, file)
            try:
                img = imread(img_path, as_gray=True)
                features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                X.append(features)
                y.append(label_idx)
            except Exception as e:
                pass 
                
    return np.array(X), np.array(y), classes

def run_svm_pipeline():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ ERRORE: Dataset non trovato in {TRAIN_DIR}.")
        return

    print("⚙️ Avvio Pipeline Classica HOG + SVM...")
    X_train, y_train, class_names = load_data_and_extract_hog(TRAIN_DIR)
    X_test, y_test, _ = load_data_and_extract_hog(TEST_DIR)

    print(f"✅ Feature estratte con successo! Dimensione matrice training: {X_train.shape}")
    print("🧠 Addestramento Support Vector Machine (LinearSVC) in corso...")
    
    model = LinearSVC(class_weight='balanced', dual=False, max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    print("📊 Generazione Predizioni e Report...")
    y_pred = model.predict(X_test)

    print("\n📝 REPORT DI CLASSIFICAZIONE HOG + SVM:")
    print(classification_report(y_test, y_pred, target_names=class_names))

if __name__ == '__main__':
    run_svm_pipeline()
