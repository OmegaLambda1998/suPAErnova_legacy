#!/usr/bin/env python
"""This code constructs and trains the normalizing flow model,
based on the parameters specified in the configuration file, config/train.yaml.

The Autoencoder architecture is specified in models/autoencoder.py,
and the loss terms are specified in models/losses.py.
"""

import os
from typing import TYPE_CHECKING
import argparse

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

from supaernova_legacy.utils import data_loader
from supaernova_legacy.models import (
    loader as model_loader,
    flow_training,
)
from supaernova_legacy.utils.YParams import YParams

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tensorflow import keras as ks
    import tensorflow_probability as tfp


def parse_arguments(inputs: "Sequence[str] | None") -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--yaml_config", default="../config/train.yaml", type=str)
    parser.add_argument("--config", default="nflow", type=str)
    parser.add_argument("--print_params", default=True, action="store_true")

    return parser.parse_args(inputs)


def train_flow(
    inputs: "Sequence[str] | None" = None,
) -> tuple["ks.Model", "tfp.distributions.TransformedDistribution", "YParams"]:
    args = parse_arguments(inputs)

    params = YParams(
        os.path.abspath(args.yaml_config), args.config, print_params=args.print_params
    )
    params["print_params"] = args.print_params

    for _il, latent_dim in enumerate(params["latent_dims"]):
        print(f"Training model with {latent_dim:d} latent dimensions")
        params["latent_dim"] = latent_dim

        encoder, _decoder, AE_params = model_loader.load_ae_models(params)

        train_data = data_loader.load_data(
            os.path.join(
                params["PROJECT_DIR"],
                params["train_data_file"],
            ),
            print_params=params["print_params"],
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
        train_data["mask_sn"] = data_loader.get_train_mask(train_data, AE_params.params)
        test_data["mask_sn"] = data_loader.get_train_mask(test_data, AE_params.params)

        # Mask certain spectra
        train_data["mask_spectra"] = data_loader.get_train_mask_spectra(
            train_data, AE_params.params
        )
        test_data["mask_spectra"] = data_loader.get_train_mask_spectra(
            test_data, AE_params.params
        )

        # Get latent representations from encoder
        train_data["z_latent"] = encoder((
            train_data["spectra"],
            train_data["times"],
            train_data["mask"] * train_data["mask_spectra"],
        )).numpy()

        test_data["z_latent"] = encoder((
            test_data["spectra"],
            test_data["times"],
            test_data["mask"] * test_data["mask_spectra"],
        )).numpy()

        train_data["z_latent"] = train_data["z_latent"][train_data["mask_sn"]]
        test_data["z_latent"] = test_data["z_latent"][test_data["mask_sn"]]

        # Split off validation set from training set
        # train_data, val_data = data_loader.split_train_and_val(train_data, params)

        checkpoint_filepath = (
            "{:s}flow_kfold{:d}_{:02d}Dlatent_layers{:s}_nlayers{:02d}_{:s}/".format(
                params["NFLOW_MODEL_DIR"],
                params["kfold"],
                params["latent_dim"],
                "-".join(str(e) for e in params["encode_dims"]),
                params["nlayers"],
                params["out_file_tail"],
            )
        )
        checkpoint_filepath = os.path.join(params["PROJECT_DIR"], checkpoint_filepath)
        print(checkpoint_filepath)
        if os.path.exists(checkpoint_filepath):
            print("Skipping nflow training")
        else:
            # Saved on checkpoint, so no need to save again
            _NFmodel, _flow = flow_training.train_flow(
                train_data,
                test_data,
                params,
            )
    return nflow_results(params)


def nflow_results(
    params: YParams,
):
    nf_model, flow = model_loader.load_flow(params)
    return (nf_model, flow, params)


if __name__ == "__main__":
    train_flow()
