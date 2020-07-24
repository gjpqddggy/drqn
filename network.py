import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from config import BATCH_SIZE

def reshape(out):
    shape = out.get_shape().as_list()
    out_flat = tf.reshape(out, [tf.shape(out)[0], 1, shape[1] * shape[2] * shape[3]])
    return out_flat

def build_q_network(n_actions, learning_rate=0.00001, input_shape=(84, 84), history_length=4):
    """Builds a dueling DQN as a Keras model
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
    Returns:
        A compiled Keras model
    """
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)  # normalize by 255
    #print(x.shape)
    AE_model = tf.keras.models.load_model("./checkpoint/AE.h5", compile=False)
    AE_model.trainable = False
    encoder = AE_model.layers[0]
    # x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    # x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    # x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    #print(x.get_shape())
    out_flat = Lambda(lambda f: reshape(f))(x)
    #print(out_flat.get_shape())
    
    lstm = tf.keras.layers.LSTM(512)
    x = lstm(out_flat)
    #print(x.get_shape())

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, num_or_size_splits=2, axis=1))(x)  # custom splitting layer

    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean

    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model
