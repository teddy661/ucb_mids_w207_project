import os

import keras_tuner as kt
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("INFO")

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96

# don't change the order of this list, to be consistent with the csv file
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

ALL_LABELS = [
    "left_eye_center",
    "right_eye_center",
    "left_eye_inner_corner",
    "left_eye_outer_corner",
    "right_eye_inner_corner",
    "right_eye_outer_corner",
    "left_eyebrow_inner_end",
    "left_eyebrow_outer_end",
    "right_eyebrow_inner_end",
    "right_eyebrow_outer_end",
    "nose_tip",
    "mouth_left_corner",
    "mouth_right_corner",
    "mouth_center_top_lip",
    "mouth_center_bottom_lip",
]


class FaceKeyPointModelTuner(kt.HyperModel):
    """
    HyperModel for the facial recognition models
    """

    def __init__(self, labels: list[str], name=None, tunable=True):
        super().__init__(name, tunable)
        self.labels = labels

    def build_model(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """
        Builds the model with the hyperparameters, used by the tuner later
        """

        tf.keras.backend.clear_session()
        input_layer = tf.keras.layers.Input(
            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="InputLayer"
        )
        rescale = tf.keras.layers.Rescaling(
            1.0 / 255, name="rescaling", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        )(input_layer)

        ## Begin Convolutional Layers
        prev_layer = rescale
        for cur_con_layer in range(hp.Int("num_conv_layers", 3, 5)):

            filter_size = hp.Int("filter_size", 32, 128, 32)
            kernel_size = hp.Int("kernel_size", 3, 5, 2)

            conv_1 = tf.keras.layers.Conv2D(
                filters=filter_size,
                kernel_size=kernel_size,
                strides=(1, 1),
                name="conv_1st_" + str(cur_con_layer),
                padding="same",
                kernel_initializer="he_uniform",
                activation="relu",
            )(prev_layer)

            conv_2 = tf.keras.layers.Conv2D(
                filters=filter_size,
                kernel_size=kernel_size,
                strides=(1, 1),
                name="conv_2nd_" + str(cur_con_layer),
                padding="same",
                kernel_initializer="he_uniform",
                activation="relu",
            )(conv_1)

            maxp = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), padding="same", name="pool_" + str(cur_con_layer)
            )(conv_2)

            # drop_1 =tf.keras.layers.Dropout(0.25, name="Dropout_1")(maxp_1)
            norm = tf.keras.layers.BatchNormalization(
                name="norm_" + str(cur_con_layer)
            )(maxp)

            prev_layer = norm

        ## Fully Connected layers

        flat_1 = tf.keras.layers.Flatten()(prev_layer)
        dense_1 = tf.keras.layers.Dense(
            hp.Int("fc1_units", 512, 1024, 512),
            name="fc_1",
            kernel_initializer="he_uniform",
            activation="elu",
        )(flat_1)
        # drop_1 =tf.keras.layers.Dropout(0.20, name="Dropout_1")(dense_1)
        norm_100 = tf.keras.layers.BatchNormalization(name="norm_100")(dense_1)

        dense_2 = tf.keras.layers.Dense(
            hp.Int("fc2_units", 256, 1024, 256),
            name="fc_2",
            kernel_initializer="he_uniform",
            activation="elu",
        )(norm_100)
        # drop_2 =tf.keras.layers.Dropout(0.20, name="Dropout_2")(dense_2)
        norm_101 = tf.keras.layers.BatchNormalization(name="norm_101")(dense_2)

        ##
        ## End Fully Connected Layers
        ##

        ## Construct Output Layers, loss and metrics
        output_layers = []
        loss_dict = {}
        metrics_dict = {}
        for i, label in enumerate(self.labels):
            # append 2 output layers (_x and _y) for each label, given the coordinates
            for j, coord in enumerate(["_x", "_y"]):
                output_layers.append(
                    tf.keras.layers.Dense(
                        units=1,
                        activation=None,
                        name=label + coord,
                    )(norm_101)
                )
                loss_dict[label + coord] = "mse"
                metrics_dict[label + coord] = "mae"

        model = tf.keras.Model(
            inputs=[input_layer],
            outputs=output_layers,
            name=self.name,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Nadam(
                learning_rate=hp.Choice("lr", 1e-4, 1e-3)
            ),
            loss=loss_dict,
            metrics=metrics_dict,
        )

        return model

    def fit(self, hp:kt.HyperParameters, model:tf.keras.models.Model, x, y, validation_data, *args, **kwargs):

        x_val, y_val = validation_data
        y = self.convert_y_to_dictonary(y)
        y_val = self.convert_y_to_dictonary(y_val)
        validation_data = (x_val, y_val)

        # define callbacks
        # based on Ed, early stopping proves to be the best option
        # not going to tune it anymore for now
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=2,
            patience=hp.Int("patience", 10, 20, 50),
            min_delta=0.0001,
            restore_best_weights=True,
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "model_checkpoints/{epoch:04d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_weights_only=False,
            save_best_only=False,
        )

        return model.fit(
            *args,
            x=x,
            y=y,
            validation_data=validation_data,
            batch_size=hp.Choice("batch_size", [16, 32, 64]),
            callbacks=[early_stopping],
            **kwargs,
        )

    def convert_y_to_dictonary(self, y_array):
        """
        Converts the y array to a dictionary, based on self.labels.
        This assumes that the y array is in the same order as self.labels.
        """

        y_dict = {}
        for i, col in enumerate(Y_COLUMN_NAMES):
            if col[:-2] in self.labels:
                y_dict[col] = y_array[:, i]
        return y_dict
                y_dict[col] = y_array[:, i]
        return y_dict
