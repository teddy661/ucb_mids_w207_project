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

BEST_JOINT_HP = {
    "num_conv_layers": 5,
    "filter_size": 64,
    "fc_units": 4096,
}

BEST_STAGE_ONE_HP = {
    "num_conv_layers": 4,
    "filter_size": 32,
    "fc_units": 2048,
}

BEST_INDIVIDUAL_HP = {
    "num_conv_layers": 4,
    "filter_size": 32,
    "fc_units": 2048,
}


class FaceKeyPointHyperModel(kt.HyperModel):
    """
    HyperModel for the facial recognition models.

    Some doc: https://keras.io/guides/keras_tuner/getting_started/#tune-model-training
    """

    def __init__(self, labels, name=None, tunable=True):
        super().__init__(name, tunable)

        self.labels = sorted(
            labels, key=lambda x: ALL_LABELS.index(x)
        )  # sort the labels based on required index

        # get the predefined hyperparameters, since Ray already tuned it before
        self.predected_hp = {}
        if self.name == "model_joint" or self.name == "model_stage_two":
            self.predected_hp = BEST_JOINT_HP
        elif self.name == "model_stage_one":
            self.predected_hp = BEST_STAGE_ONE_HP
        elif len(self.labels) == 1:
            self.predected_hp = BEST_INDIVIDUAL_HP

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """
        Builds the model with the hyperparameters, used by the tuner later
        """

        tf.keras.backend.clear_session()
        input_layer = tf.keras.layers.Input(
            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="InputLayer"
        )

        flat_1 = self.build_cnn_layers(hp, input_layer)
        ## Fully Connected layers

        fc_units = (
            self.predected_hp["fc_units"]
            if "fc_units" in self.predected_hp
            else hp.Choice("fc_units", [2048, 4096, 8192])
        )

        dense_1 = tf.keras.layers.Dense(
            fc_units,
            name="dense_1",
            kernel_initializer="he_uniform",
            activation="relu",
        )(flat_1)
        norm_fc1 = tf.keras.layers.BatchNormalization(name="norm_fc1")(dense_1)

        # after tuning, we found that fc units should always decrease
        fc_units = fc_units // 2

        dense_2 = tf.keras.layers.Dense(
            fc_units,
            name="dense_2",
            kernel_initializer="he_uniform",
            activation="relu",
        )(norm_fc1)
        norm_fc2 = tf.keras.layers.BatchNormalization(name="norm_fc2")(dense_2)

        ## Construct Output Layers, loss and metrics
        output_layers = []
        loss_dict = {}
        metrics_dict = {}
        for col_name in self.get_column_names():
            output_layers.append(
                tf.keras.layers.Dense(
                    units=1,
                    activation=None,
                    name=col_name,
                )(norm_fc2)
            )
            loss_dict[col_name] = "mse"
            metrics_dict[col_name] = "mse"

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

    def build_cnn_layers(self, hp: kt.HyperParameters, input_layer: tf.keras.layers):

        rescale = tf.keras.layers.Rescaling(
            1.0 / 255,
            name="rescaling",
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1),
        )(input_layer)

        ## Begin Convolutional Layers
        total_num_conv_layers = (
            self.predected_hp["num_conv_layers"]
            if "num_conv_layers" in self.predected_hp
            else hp.Int("num_conv_layers", 4, 6)
        )

        prev_layer = rescale
        for cur_con_layer in range(total_num_conv_layers):

            filter_size = (
                self.predected_hp["filter_size"]
                if "filter_size" in self.predected_hp
                else hp.Int("filter_size", 32, 64, 32)
            )
            kernel_size = (3, 3)
            if filter_size < 256:
                filter_size *= 2
            num_of_conv_layers = 2 if cur_con_layer >= 2 else 2

            for i in range(num_of_conv_layers):
                conv_1 = tf.keras.layers.Conv2D(
                    filters=filter_size,
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    name=f"conv_{str(i)}_{str(cur_con_layer)}",
                    padding="same",
                    kernel_initializer="he_uniform",
                    activation="relu",
                )(prev_layer)

                norm_1 = tf.keras.layers.BatchNormalization(
                    name=f"norm_{str(i)}_{str(cur_con_layer)}",
                )(conv_1)
                prev_layer = norm_1

            maxp = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                padding="same",
                name=f"mpool_{str(cur_con_layer)}",
            )(prev_layer)

            # after tuning, we found that the dropout is not needed
            # if hp.Boolean("dropout"):
            #     dropout = tf.keras.layers.Dropout(
            #         rate=0.25,
            #         name= f"dropout_{str(cur_con_layer)}",
            #     )(maxp)
            #     prev_layer = dropout
            # else:
            prev_layer = maxp

        flat_1 = tf.keras.layers.Flatten(name="flat")(prev_layer)

        return flat_1

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
        """
        Fits the model with the hyperparameters, used by the tuner later.
        """
        # We are done tuning with parameters here, so just pass the data along

        # add the custom callbacks on top of the tuner's callbacks
        callbacks = kwargs["callbacks"]
        callbacks.extend(self.get_callbacks(hp))

        return model.fit(
            *args,
            x=x,
            y=y,
            validation_data=validation_data,
            batch_size=32,  # hp.Choice("batch_size", [32, 64]),
            **kwargs,
        )

    def get_column_names(self) -> list[str]:
        """
        Returns the column names of the output layers
        """
        return [label + coord for label in self.labels for coord in ["_x", "_y"]]

    def convert_y_to_outputs(self, y_array):
        """
        Converts the y array to a dictionary, based on self.labels.
        This assumes that the y array is in the same order as self.labels.
        """

        y_dict = {}
        # have to loop through ALL_Y_COLUMNS, as the order might be wrong in self.labels
        for i, col in enumerate(self.get_column_names()):
            y_dict[col] = y_array[:, i]

        return y_dict

    def get_callbacks(self, hp: kt.HyperParameters):

        callbacks = []
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=2,
            patience=20,
            min_delta=0.0001,
            restore_best_weights=True,
        )
        callbacks.append(early_stopping)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "model_checkpoints/{epoch:04d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            verbose=2,
            save_weights_only=False,
            save_best_only=False,
        )
        # callbacks.append(model_checkpoint)

        reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=5, verbose=2, factor=0.3, min_lr=10e-7
        )
        callbacks.append(reduce_lr_on_plateau)

        return callbacks


class FaceKeyPointStageTwoHM(FaceKeyPointHyperModel):
    def __init__(self, labels, name=None, tunable=True):
        super().__init__(labels, name, tunable)

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """
        Builds the model with the hyperparameters, used by the tuner later.
        This is the second stage of the model, where we use the output of the first stage
        """

        tf.keras.backend.clear_session()

        input_layer_stage_1 = tf.keras.layers.Input(shape=(8), name="input_stage_1")
        norm_stage_1 = tf.keras.layers.BatchNormalization(name="norm_stage_1")(
            input_layer_stage_1
        )
        stage_1_dense_1 = tf.keras.layers.Dense(
            4,
            name="dense_1_stage_1",
            kernel_initializer="he_uniform",
            activation="relu",
        )(norm_stage_1)
        last_layer_stage_1 = stage_1_dense_1

        input_layer_stage_2 = tf.keras.layers.Input(
            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="input_stage_2"
        )
        flat_stage_2 = self.build_cnn_layers(hp, input_layer_stage_2)

        # connect two layers together from stage 1 and stage 2
        concatenate_layer = tf.keras.layers.Concatenate()(
            [last_layer_stage_1, flat_stage_2]
        )

        fc_units = (
            self.predected_hp["fc_units"]
            if "fc_units" in self.predected_hp
            else hp.Choice("fc_units", [2048, 4096, 8192])
        )
        dense_1 = tf.keras.layers.Dense(
            fc_units,
            name="dense_1",
            kernel_initializer="he_uniform",
            activation="relu",
        )(concatenate_layer)
        norm_fc1 = tf.keras.layers.BatchNormalization(name="norm_fc1")(dense_1)

        # after tuning, we found that fc units should always decrease
        fc_units = fc_units // 2

        dense_2 = tf.keras.layers.Dense(
            fc_units,
            name="dense_2",
            kernel_initializer="he_uniform",
            activation="relu",
        )(norm_fc1)
        norm_fc2 = tf.keras.layers.BatchNormalization(name="norm_fc2")(dense_2)

        ## Construct Output Layers, loss and metrics
        output_layers = []
        loss_dict = {}
        metrics_dict = {}
        for col_name in self.get_column_names():
            output_layers.append(
                tf.keras.layers.Dense(
                    units=1,
                    activation=None,
                    name=col_name,
                )(norm_fc2)
            )
            loss_dict[col_name] = "mse"
            metrics_dict[col_name] = "mse"

        model = tf.keras.Model(
            inputs=[input_layer_stage_1, input_layer_stage_2],
            outputs=output_layers,
            name=self.name,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss_dict,
            metrics=metrics_dict,
        )

        return model
