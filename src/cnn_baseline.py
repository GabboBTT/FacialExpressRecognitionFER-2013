import tensorflow as tf
from tensorflow.keras import layers, models

def build_mini_xception(input_shape=(48, 48, 1), num_classes=7):
    """
    Versione completa della Mini-Xception per FER-2013.
    """
    # --- DATA AUGMENTATION INTEGRATA ---
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)

    # Blocco 1
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Blocco 2 (Separable)
    residual = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Blocco 3 (Separable)
    residual = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Classificatore
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x) # Previene l'overfitting sui pesi bilanciati
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='Mini_Xception_Full')
    return model

if __name__ == "__main__":
    model = build_mini_xception()
    model.summary()