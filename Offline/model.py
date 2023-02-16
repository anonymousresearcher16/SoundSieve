import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape, num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(1, kernel_size=(1,4), activation="relu", strides=(1,4)),
        layers.Conv2D(2, kernel_size=(3,3), activation="relu", strides=(2,1)), #has to be 2
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(8, kernel_size=(3,3), activation="relu"), #has to be 8
        layers.MaxPooling2D(pool_size=(2,2)),        
        layers.Conv2D(32, kernel_size=(3,3), activation="relu"), #16
        layers.GlobalMaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model

model = create_model((60,128,1), 8)
print(model.summary())