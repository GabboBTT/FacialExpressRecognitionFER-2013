import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
NUM_CLASSES = 7

def build_mini_xception(input_shape=(48, 48, 1), num_classes=7):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    residual = layers.Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    
    x = layers.SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='Mini_Xception_Baseline')
    return model

if __name__ == "__main__":
    model = build_mini_xception()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])
    print("Sanity Check superato: Architettura compilata correttamente.")
