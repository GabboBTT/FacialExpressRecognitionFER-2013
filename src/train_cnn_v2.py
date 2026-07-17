import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# Parametri
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
TRAIN_DIR = '../dataset/train'
VAL_DIR = '../dataset/test'

def build_and_train_v2():
    # 1. Pesi Bilanciati
    classes = sorted(os.listdir(TRAIN_DIR))
    y = [i for i, cls in enumerate(classes) for _ in os.listdir(os.path.join(TRAIN_DIR, cls))]
    class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y), y=y)))

    # 2. Caricamento Dataset in Scala di Grigi
    train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, color_mode='grayscale', image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical')
    val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR, color_mode='grayscale', image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical')

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # 3. Data Augmentation Dinamica
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1)
    ])

    # 4. Architettura CNN V2
    inputs = layers.Input(shape=(48, 48, 1))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    # Blocchi Convoluzionali
    for filters in [64, 128, 256]:
        x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.25)(x)

    # Top Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(7, activation='softmax')(x)

    model_v2 = models.Model(inputs, outputs)

    # 5. Compile & Train
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model_v2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])

    print("🔥 Avvio Addestramento CNN V2...")
    # model_v2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights)

if __name__ == '__main__':
    build_and_train_v2()
