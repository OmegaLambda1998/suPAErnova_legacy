#!/usr/bin/env python
"""This code performs posterior analysis,
based on the parameters specified in the configuration file, config/posterior_analysis.yaml.

To find the maximum of the posterior (MAP) we begin LBFGS optimization from the best fit encoded value of the data, as well as additional randomly initialized points in the parameter space. We denote the MAP latent variables as the best fit parameters that maximize the posterior from these minima. From the MAP value we then run Hamiltonian Monte Carlo (HMC) to marginalize over the parameters to obtain the final best fit model parameters and their uncertainty.

The Autoencoder architecture is specified in models/autoencoder.py,
The flow architecture is specified in models/flow.py,
"""

import os
from typing import TYPE_CHECKING
import argparse

from supaernova_legacy.utils import data_loader, calculations
from supaernova_legacy.models import (
    loader as model_loader,
    posterior_analysis,
)
from supaernova_legacy.utils.YParams import YParams

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence

# def find_MAP(model, params, verbose=False):

# def run_HMC(model, params, verbose=False):

# def train(PAE, params, train_data, test_data, tstrs=['train', 'test']):


def parse_arguments(inputs: "Sequence[str] | None") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_config", default="../config/posterior_analysis.yaml", type=str
    )
    parser.add_argument("--config", default="posterior", type=str)
    parser.add_argument("--print_params", default=True, action="store_true")

    return parser.parse_args(inputs)


def run_posterior_analysis(inputs: "Sequence[str] | None" = None) -> None:
    args = parse_arguments(inputs)

    params = YParams(os.path.abspath(args.yaml_config), args.config, print_params=True)

    results = {}

    for il, latent_dim in enumerate(params["latent_dims"]):
        print(f"Training model with {latent_dim:d} latent dimensions")
        params["latent_dim"] = latent_dim

        # Get PAE model
        PAE = model_loader.PAE(params)

        train_data = data_loader.load_data(
            os.path.join(params["PROJECT_DIR"], params["train_data_file"]),
            print_params=params["print_params"],
            set_data_min_val=params["set_data_min_val"],
            npz=True,
        )
        test_data = data_loader.load_data(
            os.path.join(params["PROJECT_DIR"], params["test_data_file"]),
            set_data_min_val=params["set_data_min_val"],
            npz=True,
        )

        # Mask certain supernovae
        train_data["mask_sn"] = data_loader.get_train_mask(train_data, params)
        test_data["mask_sn"] = data_loader.get_train_mask(test_data, params)

        # Mask certain spectra
        train_data["mask_spectra"] = data_loader.get_train_mask_spectra(
            train_data, params
        )
        test_data["mask_spectra"] = data_loader.get_train_mask_spectra(
            test_data, params
        )

        train_data["mask"] *= train_data["mask_spectra"]
        test_data["mask"] *= test_data["mask_spectra"]

        # Get latent representations from encoder and flow
        train_data["z_latent"] = PAE.encoder((
            train_data["spectra"],
            train_data["times"],
            train_data["mask"],
        )).numpy()
        test_data["z_latent"] = PAE.encoder((
            test_data["spectra"],
            test_data["times"],
            test_data["mask"],
        )).numpy()

        istart = 0
        if params["physical_latent"]:
            istart = 2
        train_data["u_latent"] = PAE.flow.bijector.inverse(
            train_data["z_latent"][:, istart:]
        ).numpy()
        test_data["u_latent"] = PAE.flow.bijector.inverse(
            test_data["z_latent"][:, istart:]
        ).numpy()

        # Get reconstructions
        train_data["spectra_ae"] = PAE.decoder((
            train_data["z_latent"],
            train_data["times"],
            train_data["mask"],
        )).numpy()
        test_data["spectra_ae"] = PAE.decoder((
            test_data["z_latent"],
            test_data["times"],
            test_data["mask"],
        )).numpy()

        # Measure AE reconstruction uncertainty as a function of time
        dm = train_data["mask_sn"]
        print(
            train_data["spectra"].shape,
            dm.shape,
            train_data["mask_sn"].shape,
            train_data["mask_spectra"].shape,
        )
        (
            train_data["sigma_ae_time"],
            _ae_noise_t_bin_edge,
            train_data["sigma_ae_time_tbin_cent"],
        ) = calculations.compute_sigma_ae_time(
            train_data["spectra"][dm],
            train_data["spectra_ae"][dm],
            train_data["sigma"][dm],
            train_data["times"][dm],
            train_data["mask"][dm],
        )

        dm = test_data["mask_sn"]
        (
            test_data["sigma_ae_time"],
            _ae_noise_t_bin_edge,
            test_data["sigma_ae_time_tbin_cent"],
        ) = calculations.compute_sigma_ae_time(
            test_data["spectra"][dm],
            test_data["spectra_ae"][dm],
            test_data["sigma"][dm],
            test_data["times"][dm],
            test_data["mask"][dm],
        )

        tstrs = ["train", "test"]
        # tstrs = ['train']
        # tstrs = ['test']
        results[il] = posterior_analysis.train(
            PAE, params, train_data, test_data, tstrs=tstrs
        )

    return posterior_results(params, results)


def posterior_results(params: YParams, results: dict[str, "Any"]):
    return (results, params)


if __name__ == "__main__":
    run_posterior_analysis()
