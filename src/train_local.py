import tensorflow as tf
from cnn_baseline import build_mini_xception

# Costanti di test
IMG_SIZE = (48, 48)
BATCH_SIZE = 2
DATA_DIR = '../data/dummy_fer/train'

def test_pipeline():
    print("Avvio Sanity Check Pipeline Dati...")
    
    try:
        # 1. Inizializzazione Data Loader
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            color_mode='grayscale',
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
        
        # ESTRAZIONE CLASSI PRIMA DELL'OTTIMIZZAZIONE
        class_names = train_ds.class_names
        num_classes = len(class_names)
        print(f"Trovate {num_classes} classi: {class_names}")
        
        # Ottimizzazione I/O
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        # 2. Caricamento Modello
        model = build_mini_xception(input_shape=(48, 48, 1), num_classes=num_classes)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        
        # 3. Test Forward Pass
        print("\nEsecuzione Forward Pass di test...")
        model.fit(train_ds, epochs=1, steps_per_epoch=1)
        
        print("\n✅ SANITY CHECK SUPERATO! Tensori allineati e pipeline funzionante.")
        
    except Exception as e:
        print(f"\n❌ ERRORE NEL SANITY CHECK: {e}")

if __name__ == "__main__":
    test_pipeline()
