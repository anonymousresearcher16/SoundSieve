import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

DATASET_NAME = "Urban"
predictor_model_name = DATASET_NAME + "/importance_checkpoint/predictor_test"

def save_model(model, model_name):
    model.save_weights(model_name + "_weights")
    model.save(model_name+".h5")
    print('Model Saved')

def get_predictor_model(input_shape, prediction_length, load=False):
    predictor_model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(prediction_length)
    ])
    if load:
        predictor_model.load_weights(predictor_model_name + "_weights")
    
    return predictor_model



def train():
    input_shape = (3, 32, 1)
    prediction_length = 5
    batch_size = 64
    lr=1e-3
    EPOCHS = 10000


    predictor_model = get_predictor_model(input_shape, prediction_length)
    
    training_data = np.load(DATASET_NAME + "/importance_training_data.npy", allow_pickle=True).item()
    X = training_data['X']
    y = training_data['y']
    print(y)
    y = (y-np.min(y))/(np.max(y)-np.min(y)) * 1e3
    
    #mask = np.random.rand(X.shape[0]) <= 0.8
    #mask = np.zeros((X.shape[0]))
    #i = 0
    #while i < X.shape[0]:
    #    mask[i:i+1] = 1
    #    i += 15
    #mask = mask > 0

    #X = X[mask]
    #y = y[mask]

    mask = np.random.rand(X.shape[0]) <= 0.8
    training_X = X[mask]
    training_y = y[mask]
    testing_X = X[~mask]
    testing_y = y[~mask]
    print(training_y.shape, testing_y.shape)

    dataset = tf.data.Dataset.from_tensor_slices((training_X, training_y)).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((testing_X, testing_y)).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.MeanSquaredError()
    #loss_fn = weighted_loss

    best_loss = 9999.99
    for epoch in range(EPOCHS):
        current_loss = 0.0
        num_batches = 0
        for batch_inputs in dataset:
            features, importance = batch_inputs

            with tf.GradientTape() as tape:
                """ Produce model predictions """
                predicted_importance = predictor_model(features, training=True)  
                if epoch > 20 and epoch%50 == 0:
                    print(predicted_importance)

                """ Compute loss value """
                loss_value = loss_fn(importance, predicted_importance)
            
            """ Calculate gradient and apply optimizer """   
            grads = tape.gradient(loss_value, predictor_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, predictor_model.trainable_variables))

            current_loss += loss_value
            num_batches += 1
        
        print(f"Training loss: {current_loss/num_batches}")

        current_loss = 0.0
        num_batches = 0
        
        for batch_inputs in test_dataset:
            features, importance = batch_inputs

            """ Produce model predictions """
            predicted_importance = predictor_model(features, training=False)      

            """ Compute loss value """
            
            loss_value = loss_fn(importance, predicted_importance)

            current_loss += loss_value
            num_batches += 1
        
        loss_val = current_loss/num_batches
        if loss_val < best_loss:
            best_loss = loss_val
            save_model(predictor_model, predictor_model_name)

        
        print(f"Testing loss: {loss_val}")

#train()
