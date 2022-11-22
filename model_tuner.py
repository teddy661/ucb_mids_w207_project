import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96


Y_COLUMN_NAMES = [
    "left_eye_center_X",
    "left_eye_center_Y",
    "right_eye_center_X",
    "right_eye_center_Y",
    "left_eye_inner_corner_X",
    "left_eye_inner_corner_Y",
    "left_eye_outer_corner_X",
    "left_eye_outer_corner_Y",
    "right_eye_inner_corner_X",
    "right_eye_inner_corner_Y",
    "right_eye_outer_corner_X",
    "right_eye_outer_corner_Y",
    "left_eyebrow_inner_end_X",
    "left_eyebrow_inner_end_Y",
    "left_eyebrow_outer_end_X",
    "left_eyebrow_outer_end_Y",
    "right_eyebrow_inner_end_X",
    "right_eyebrow_inner_end_Y",
    "right_eyebrow_outer_end_X",
    "right_eyebrow_outer_end_Y",
    "nose_tip_X",
    "nose_tip_Y",
    "mouth_left_corner_X",
    "mouth_left_corner_Y",
    "mouth_right_corner_X",
    "mouth_right_corner_Y",
    "mouth_center_top_lip_X",
    "mouth_center_top_lip_Y",
    "mouth_center_bottom_lip_X",
    "mouth_center_bottom_lip_Y",
]

# variables to define
model_name = "model"
point_names = list(set([y_column_name[:-2] for y_column_name in Y_COLUMN_NAMES]))


def convert_y_to_dictonary(y_nd):
    """Converts the y array to a dictionary"""

    y_dict = {}
    for i, col in enumerate(Y_COLUMN_NAMES):
        y_dict[col] = y_nd[:, i]

    return y_dict


class HyperModelTuner(kt.HyperModel):
    """
    HyperModel for the facial recognition models
    """

    def build_model(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """
        Builds the model with the hyperparameters, used by the tuner later
        """

        tf.keras.backend.clear_session()
        input_layer = keras.layers.Input(
            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="InputLayer"
        )
        rescale = keras.layers.Rescaling(
            1.0 / 255, name="rescaling", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        )(input_layer)

        ## Begin Convolutional Layers
        prev_layer = rescale
        for cur_con_layer in range(hp.Int("num_conv_layers", 3, 5)):

            filter_size = hp.Int("filter_size", 32, 128, 32)
            kernel_size = hp.Int("kernel_size", 3, 5, 2)

            conv_1 = keras.layers.Conv2D(
                filters=filter_size,
                kernel_size=kernel_size,
                strides=(1, 1),
                name="conv_1st_" + str(cur_con_layer),
                padding="same",
                kernel_initializer="he_uniform",
                activation="relu",
            )(prev_layer)

            conv_2 = keras.layers.Conv2D(
                filters=filter_size,
                kernel_size=kernel_size,
                strides=(1, 1),
                name="conv_2nd_" + str(cur_con_layer),
                padding="same",
                kernel_initializer="he_uniform",
                activation="relu",
            )(conv_1)

            maxp = keras.layers.MaxPooling2D(
                pool_size=(2, 2), padding="same", name="pool_" + str(cur_con_layer)
            )(conv_2)

            # drop_1 = keras.layers.Dropout(0.25, name="Dropout_1")(maxp_1)
            norm = keras.layers.BatchNormalization(name="norm_" + str(cur_con_layer))(
                maxp
            )

            prev_layer = norm

        ## Fully Connected layers

        flat_1 = keras.layers.Flatten()(prev_layer)
        dense_1 = tf.keras.layers.Dense(
            hp.Int("fc1_units", 512, 1024, 512),
            name="fc_1",
            kernel_initializer="he_uniform",
            activation="elu",
        )(flat_1)
        # drop_1 = keras.layers.Dropout(0.20, name="Dropout_1")(dense_1)
        norm_100 = keras.layers.BatchNormalization(name="norm_100")(dense_1)

        dense_2 = keras.layers.Dense(
            hp.Int("fc2_units", 256, 1024, 256),
            name="fc_2",
            kernel_initializer="he_uniform",
            activation="elu",
        )(norm_100)
        # drop_2 = keras.layers.Dropout(0.20, name="Dropout_2")(dense_2)
        norm_101 = keras.layers.BatchNormalization(name="norm_101")(dense_2)

        ##
        ## End Fully Connected Layers
        ##

        ## Construct Output Layers, loss and metrics
        output_layers = []
        loss_dict = {}
        metrics_dict = {}
        for i, col in enumerate(Y_COLUMN_NAMES):
            output_layers.append(
                keras.layers.Dense(units=1, activation=None, name=col)(norm_101)
            )
            loss_dict[col] = "mse"
            metrics_dict[col] = "mse"

        model = tf.keras.Model(
            inputs=[input_layer],
            outputs=output_layers,
            name="FacialKeypoints",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001),
            loss=loss_dict,
            metrics=metrics_dict,
        )

        return model

    def fit(self, hp, model, x, y, validation_data, *args, **kwargs):

        y = convert_y_to_dictonary(y)
        x_val, y_val = validation_data
        y_val = convert_y_to_dictonary(y_val)
        validation_data = (x_val, y_val)

        return model.fit(
            *args,
            x=x,
            y=y,
            validation_data=validation_data,
            batch_size=hp.Choice("batch_size", [16, 32, 64]),
            **kwargs,
        )
