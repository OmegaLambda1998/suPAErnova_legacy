# !/usr/bin/env python
"""This code constructs and trains the Autoencoder model, based on the parameters specified in the configuration file, config/train.yaml.

The Autoencoder architecture is specified in models/autoencoder.py,
and the loss terms are specified in models/losses.py.
"""

import os
from typing import TYPE_CHECKING
import argparse

import tensorflow as tf

from supaernova_legacy.utils import data_loader
from supaernova_legacy.models import (
    loader as model_loader,
    autoencoder,
    autoencoder_training,
)
from supaernova_legacy.utils.YParams import YParams

if TYPE_CHECKING:
    from collections.abc import Sequence

print("tensorflow version: ", tf.__version__)
print("devices: ", tf.config.list_physical_devices("GPU"))


def parse_arguments(inputs: "Sequence[str] | None") -> argparse.Namespace:
    # Set model Architecture and training params and train
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="../config/train.yaml", type=str)
    parser.add_argument("--config", default="pae", type=str)

    return parser.parse_args(inputs)


def train_ae(
    inputs: "Sequence[str] | None" = None,
) -> dict[str, tuple[autoencoder.AutoEncoder, YParams]]:
    results: dict[str, tuple[autoencoder.AutoEncoder, YParams]] = {}

    args = parse_arguments(inputs)

    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)

    train_data = data_loader.load_data(
        os.path.join(
            params["PROJECT_DIR"],
            params["train_data_file"],
        ),
        set_data_min_val=params["set_data_min_val"],
        npz=True,
    )

    test_data = data_loader.load_data(
        os.path.join(
            params["PROJECT_DIR"],
            params["test_data_file"],
        ),
        set_data_min_val=params["set_data_min_val"],
        npz=True,
    )

    # Mask certain supernovae
    train_data["mask_sn"] = data_loader.get_train_mask(train_data, params)
    test_data["mask_sn"] = data_loader.get_train_mask(test_data, params)

    # Mask certain spectra
    train_data["mask_spectra"] = data_loader.get_train_mask_spectra(train_data, params)
    test_data["mask_spectra"] = data_loader.get_train_mask_spectra(test_data, params)

    # Split off validation set from training set
    if params["use_val"]:
        train_data, val_data = data_loader.split_train_and_val(train_data, params)
    else:
        val_data = test_data

    params["train_data"] = train_data
    params["test_data"] = test_data
    params["val_data"] = val_data

    for il, latent_dim in enumerate(params["latent_dims"]):
        params["latent_dim"] = latent_dim
        params["num_training_stages"] = latent_dim + 3
        params["train_stage"] = 0

        if latent_dim == 0:
            # Model parameters are (\Delta t, \Delta m, \Delta A_v)
            # train \Delta m and \Delta A_v first. Then \Delta t
            params["train_stage"] = 1
            params["num_training_stages"] = 2

        tf.random.set_seed(params["seed"])

        # Create model
        AEmodel = autoencoder.AutoEncoder(params, training=True)

        # Model Summary
        if params["model_summary"] and (il == 0):
            print("Encoder Summary")
            AEmodel.encoder.summary()

            print("Decoder Summary")
            AEmodel.decoder.summary()

        print(f"Training model with {latent_dim:d} latent dimensions")
        # Train model, splitting into seperate training stages for seperate model parameters, if desired.

        model_save_path = os.path.join(
            params["PROJECT_DIR"], params["MODEL_DIR"], str(params["train_stage"])
        )
        param_save_path = os.path.join(
            params["PROJECT_DIR"], params["PARAM_DIR"], str(params["train_stage"])
        )

        if os.path.exists(model_save_path) and os.path.exists(param_save_path):
            print("Skipping training stage ", params["train_stage"])
        else:
            print("Running training stage ", params["train_stage"])
            _training_loss, _val_loss, _test_loss = autoencoder_training.train_model(
                train_data,
                val_data,
                test_data,
                AEmodel,
            )

        params["prev_train_stage"] = params["train_stage"]
        results[str(params["train_stage"] + 1)] = stage_results(params)
        params["train_stage"] += 1

        if not params["train_latent_individual"]:
            params["train_stage"] += params["latent_dim"] - 1

        while params["train_stage"] < params["num_training_stages"]:
            AEmodel_second = autoencoder.AutoEncoder(params, training=True)
            if params["train_stage"] < params["num_training_stages"] - 2:
                AEmodel_second.params["epochs"] = params["epochs_latent"]
            if (
                params["train_stage"] >= params["num_training_stages"] - 2
            ):  # add in delta mag
                AEmodel_second.params["epochs"] = params["epochs_final"]

            # Load best checkpoint from step 0 training
            encoder, decoder, _AE_params = model_loader.load_ae_models(params)

            final_dense_layer = len(params["encode_dims"]) + 4

            final_layer_weights = encoder.layers[final_dense_layer].get_weights()[0]
            final_layer_weights_init = AEmodel_second.encoder.layers[
                final_dense_layer
            ].get_weights()[0]

            if params["train_stage"] <= params["latent_dim"]:  # add in z_1, ..., z_n
                idim = 2 + params["train_stage"]
                final_layer_weights[:, idim] = final_layer_weights_init[:, idim] / 100

            if (
                params["train_stage"] == params["num_training_stages"] - 2
            ):  # add in delta mag
                final_layer_weights[:, 1] = final_layer_weights_init[:, 1] / 100
                if not params["train_latent_individual"]:
                    final_layer_weights[:, 3:] = final_layer_weights_init[:, 3:] / 100

            if (
                params["train_stage"] == params["num_training_stages"] - 1
            ):  # add in delta t
                final_layer_weights[:, 0] = final_layer_weights_init[:, 0] / 100

            encoder.layers[final_dense_layer].set_weights([final_layer_weights])

            AEmodel_second.encoder.set_weights(encoder.get_weights())
            AEmodel_second.decoder.set_weights(decoder.get_weights())

            model_save_path = os.path.join(
                params["PROJECT_DIR"], params["MODEL_DIR"], str(params["train_stage"])
            )
            param_save_path = os.path.join(
                params["PROJECT_DIR"], params["PARAM_DIR"], str(params["train_stage"])
            )

            if os.path.exists(model_save_path) and os.path.exists(param_save_path):
                print("Skipping training stage ", params["train_stage"])
            else:
                print("Running training stage ", params["train_stage"])
                _training_loss, _val_loss, _test_loss = (
                    autoencoder_training.train_model(
                        train_data,
                        val_data,
                        test_data,
                        AEmodel_second,
                    )
                )

            params["prev_train_stage"] = params["train_stage"]
            results[str(params["train_stage"] + 1)] = stage_results(params)
            params["train_stage"] += 1

    return results


def stage_results(params: YParams) -> tuple[autoencoder.AutoEncoder, YParams]:
    ae_model = autoencoder.AutoEncoder(params, training=False)
    encoder, decoder, ae_params = model_loader.load_ae_models(params)
    ae_model.encoder.set_weights(encoder.get_weights())
    ae_model.decoder.set_weights(decoder.get_weights())
    ae_model.bn_moving_means = ae_params["moving_means"]
    return (ae_model, ae_params)


if __name__ == "__main__":
    train_ae()
