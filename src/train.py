import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from cnn_baseline import build_mini_xception

# --- PARAMETRI PRODUZIONE (OTTIMIZZATI PER GPU) ---
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
TRAIN_DIR = '../dataset/train'
VAL_DIR = '../dataset/test'

def get_class_weights(dataset_dir):
    """Calcola i pesi bilanciati per contrastare lo sbilanciamento di FER-2013."""
    classes = sorted(os.listdir(dataset_dir))
    y = []
    for i, cls_name in enumerate(classes):
        num_images = len(os.listdir(os.path.join(dataset_dir, cls_name)))
        y.extend([i] * num_images)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    return dict(enumerate(class_weights))

def run_training():
    print("🚀 Inizializzazione Training su FER-2013...")
    
    # 1. Caricamento Dati Reali
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ ERRORE: Cartella {TRAIN_DIR} non trovata. Assicurati di aver scaricato il dataset.")
        return

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, color_mode='grayscale', image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, label_mode='categorical'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, color_mode='grayscale', image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, label_mode='categorical'
    )
    
    num_classes = len(train_ds.class_names)
    
    # Ottimizzazione I/O
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # 2. Mitigazione Bias: Class Weights
    weights = get_class_weights(TRAIN_DIR)
    print(f"⚖️ Pesi di classe bilanciati calcolati: {weights}")
    
    # 3. Costruzione e Compilazione
    model = build_mini_xception(input_shape=(48, 48, 1), num_classes=num_classes)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1) # Mitigazione Label Rumorose
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    
    # 4. Callbacks: Salva il tuo lavoro e proteggi la quota Colab
    os.makedirs('../models', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='../models/mini_xception_best.h5', monitor='val_accuracy', 
            save_best_only=True, verbose=1
        )
    ]
    
    # 5. Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=weights,
        callbacks=callbacks
    )
    print("✅ Addestramento completato e pesi migliori salvati.")

if __name__ == "__main__":
    run_training()