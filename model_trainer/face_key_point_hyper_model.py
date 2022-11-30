import os

import keras_tuner as kt
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("INFO")
tf.random.set_seed(1234)

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96

# don't change the order of this list, to be consistent with the csv file
ALL_Y_COLUMNS = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "left_eye_inner_corner_x",
    "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",
    "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",
    "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",
    "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",
    "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",
    "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",
    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",
    "right_eyebrow_outer_end_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y",
    "mouth_center_top_lip_x",
    "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
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


class FaceKeyPointHyperModel(kt.HyperModel):
    """
    HyperModel for the facial recognition models.

    Some doc: https://keras.io/guides/keras_tuner/getting_started/#tune-model-training
    """

    def __init__(self, labels, name=None, tunable=True):
        super().__init__(name, tunable)
        self.labels: list[str] = labels

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
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
        num_conv_layers = 5  # hp.Int("num_conv_layers", 3, 5)
        for cur_con_layer in range(num_conv_layers):

            # some defaults after tuning
            filter_size = 32  # hp.Int("filter_size", 32, 64, 32)
            kernel_size = (3, 3)
            # if hp.Boolean("increasing_filter_size"):
            if filter_size < 256:
                filter_size *= 2

            conv_1 = tf.keras.layers.Conv2D(
                filters=filter_size,
                kernel_size=kernel_size,
                strides=(1, 1),
                name=f"conv_1st_{str(cur_con_layer)}",
                padding="same",
                kernel_initializer="he_uniform",
                activation="relu",
            )(prev_layer)

            norm_1 = tf.keras.layers.BatchNormalization(
                name=f"norm_1st_{str(cur_con_layer)}",
            )(conv_1)

            conv_2 = tf.keras.layers.Conv2D(
                filters=filter_size,
                kernel_size=kernel_size,
                strides=(1, 1),
                name=f"conv_2nd_{str(cur_con_layer)}",
                padding="same",
                kernel_initializer="he_uniform",
                activation="relu",
            )(norm_1)

            norm_2 = tf.keras.layers.BatchNormalization(
                name=f"norm_2nd_{str(cur_con_layer)}",
            )(conv_2)

            maxp = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                padding="same",
                name=f"mpool_{str(cur_con_layer)}",
            )(norm_2)

            if hp.Boolean("dropout"):
                dropout = tf.keras.layers.Dropout(
                    rate=0.25,
                    name=f"dropout_{str(cur_con_layer)}",
                )(maxp)
                prev_layer = dropout
            else:
                prev_layer = maxp

        ## Fully Connected layers

        flat_1 = tf.keras.layers.Flatten()(prev_layer)

        fc_units = 2048  # hp.Choice("fc_units", [1024, 2048, 4096])
        dense_1 = tf.keras.layers.Dense(
            fc_units,
            name="fc_1",
            kernel_initializer="he_uniform",
            activation="relu",
        )(flat_1)
        norm_fc1 = tf.keras.layers.BatchNormalization(name="norm_fc1")(dense_1)

        # if hp.Boolean("decreasing_fc_units"):
        #     fc_units = fc_units // 2

        dense_2 = tf.keras.layers.Dense(
            fc_units,
            name="fc_2",
            kernel_initializer="he_uniform",
            activation="relu",
        )(norm_fc1)
        norm_fc2 = tf.keras.layers.BatchNormalization(name="norm_fc2")(dense_2)

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
                    )(norm_fc2)
                )
                loss_dict[label + coord] = "mse"
                metrics_dict[label + coord] = "mse"

        model = tf.keras.Model(
            inputs=[input_layer],
            outputs=output_layers,
            name=self.name,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss_dict,
            metrics=metrics_dict,
        )

        return model

    def fit(
        self,
        hp: kt.HyperParameters,
        model: tf.keras.models.Model,
        x,
        y,
        validation_data,
        *args,
        **kwargs,
    ):

        x_val, y_val = validation_data
        y_to_use = self.convert_y_to_dictonary(y)
        y_val_to_use = self.convert_y_to_dictonary(y_val)
        validation_data_to_use = (x_val, y_val_to_use)

        return model.fit(
            *args,
            x=x,
            y=y_to_use,
            validation_data=validation_data_to_use,
            batch_size=32,  # hp.Choice("batch_size", [32, 64]),
            **kwargs,
        )

    def convert_y_to_dictonary(self, y_array):
        """
        Converts the y array to a dictionary, based on self.labels.
        This assumes that the y array is in the same order as self.labels.
        """

        y_dict = {}
        col_idx = 0
        # have to loop through ALL_Y_COLUMNS, as the order might be wrong in self.labels
        for i, col in enumerate(ALL_Y_COLUMNS):
            if col[:-2] in self.labels:
                y_dict[col] = y_array[:, col_idx]
                col_idx += 1

        return y_dict
