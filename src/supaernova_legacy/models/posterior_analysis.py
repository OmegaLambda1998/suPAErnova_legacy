#!/usr/bin/env python
"""To find the maximum of the posterior (MAP) we begin LBFGS optimization from the best fit encoded value of the data, as well as additional randomly initialized points in the parameter space. We denote the MAP latent variables as the best fit parameters that maximize the posterior from these minima. From the MAP value we then run Hamiltonian Monte Carlo (HMC) to marginalize over the parameters to obtain the final best fit model parameters and their uncertainty.

The Autoencoder architecture is specified in autoencoder.py,
The flow architecture is specified in flows.py,
"""

import os
import time
from typing import TYPE_CHECKING
from pathlib import Path

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import posterior

if TYPE_CHECKING:
    from typing import Any


def find_MAP(model, params, savepath):
    ind_amplitude = 0
    ind_dtime = 0
    if params["train_dtime"]:
        ind_amplitude = 1

    for ichain in range(params["nchains"]):
        savepath_i = savepath / str(ichain)
        if (savepath_i / model.ckpt_path).exists():
            print(f"Loading posterior model from {savepath_i}")

            model.load_checkpoint(savepath_i)

            chain_min = tf.convert_to_tensor(model.map_results["chain_min"]).numpy()
            converged = tf.convert_to_tensor(model.map_results["converged"]).numpy()
            num_evaluations = tf.convert_to_tensor(
                model.map_results["num_evaluations"]
            ).numpy()
            num_chain_evaluations = tf.convert_to_tensor(
                model.map_results["num_chain_evaluations"]
            ).numpy()
            negative_log_likelihood = tf.convert_to_tensor(
                model.map_results["negative_log_likelihood"]
            ).numpy()
            amplitude = tf.convert_to_tensor(model.map_results["amplitude"]).numpy()
            dtime = tf.convert_to_tensor(model.map_results["dtime"]).numpy()
            MAPu = tf.convert_to_tensor(model.map_results["MAPu"]).numpy()
            MAPz = tf.convert_to_tensor(model.map_results["MAPz"]).numpy()
            amplitude_ini = tf.convert_to_tensor(
                model.map_results["amplitude_ini"]
            ).numpy()
            dtime_ini = tf.convert_to_tensor(model.map_results["dtime_ini"]).numpy()
            MAPu_ini = tf.convert_to_tensor(model.map_results["MAPu_ini"]).numpy()
            MAPz_ini = tf.convert_to_tensor(model.map_results["MAPz_ini"]).numpy()
            initial_position = tf.convert_to_tensor(
                model.map_results["initial_position"]
            ).numpy()

        else:
            # Run optimization from different starting points, and keep the one with lowest negative log likelihood
            print(f"Running chain: {ichain:d}")

            if ichain == 0:
                initial_position = model.MAPu_ini.numpy()
                if params["train_amplitude"]:
                    # add amplitude as first parameter
                    initial_position = np.c_[
                        model.amplitude_ini.numpy(), initial_position
                    ]
                if params["train_dtime"]:
                    # add delta time as last parameter
                    initial_position = np.c_[model.dtime_ini.numpy(), initial_position]

            if ichain > 1 and ichain < 10:
                initial_position = (
                    model.get_latent_prior().sample(model.nsamples).numpy()
                )
                if params["train_amplitude"]:
                    # add amplitude as first parameter
                    initial_position = np.c_[
                        model.get_amplitude_prior().sample(model.nsamples).numpy(),
                        initial_position,
                    ]
                if params["train_dtime"]:
                    # add delta time as last parameter
                    initial_position = np.c_[
                        model.get_dtime_prior().sample(model.nsamples).numpy(),
                        initial_position,
                    ]

            if ichain >= 10 and ichain < 20:
                # initial_position = model.get_latent_prior().sample(model.nsamples).numpy()
                initial_position = (
                    model.get_latent_prior().sample(model.nsamples).numpy() * 0.0
                )
                if params["train_amplitude"]:
                    # replace amplitude parameter with larger variance
                    # initial_position[:, 0] = model.get_amplitude_prior().sample(model.nsamples).numpy()
                    Amax = 1.5
                    Amin = -1.5
                    dA = (Amax - Amin) / (10 - 1)
                    A = (
                        np.zeros(initial_position.shape[0], dtype=np.float32)
                        + Amin
                        + (ichain - 10) * dA
                    )

                    # add amplitude as first parameter
                    initial_position = np.c_[A, initial_position]

                if params["train_dtime"]:
                    initial_position = np.c_[
                        model.get_dtime_prior().sample(model.nsamples).numpy(),
                        initial_position,
                    ]

            if ichain >= 20:
                # vary Av
                # get mean spectra in u
                initial_position = (
                    model.get_latent_prior().sample(model.nsamples).numpy() * 0.0
                )
                # transform to z
                initial_position = model.flow.bijector.forward(initial_position).numpy()

                # replace Av paramater with larger variance
                Avmax = 0.5
                Avmin = -0.5
                dA = (Avmax - Avmin) / (params["nchains"] - 20)
                Av = (
                    np.zeros(initial_position.shape[0], dtype=np.float32)
                    + Avmin
                    + (ichain - 20) * dA
                )

                initial_position[:, 0] = Av

                # transform back to u
                initial_position = model.flow.bijector.inverse(initial_position)

                # add amplitude as first parameter
                A = np.zeros(initial_position.shape[0], dtype=np.float32)
                initial_position = np.c_[A, initial_position]

                if params["train_dtime"]:
                    initial_position = np.c_[
                        model.get_dtime_prior().sample(model.nsamples).numpy(),
                        initial_position,
                    ]

            if params["train_dtime"]:
                initial_position[:, ind_dtime] *= params["dtime_norm"]

            @tf.function
            def func_bfgs(x):
                return tfp.math.value_and_gradient(lambda x: -1.0 / 100 * model(x), x)

            results = tfp.optimizer.lbfgs_minimize(
                func_bfgs,
                initial_position=initial_position,
                tolerance=params["tolerance"],
                x_tolerance=params["tolerance"],
                max_iterations=params["max_iterations"],
                num_correction_pairs=1,
            )  # ,
            # max_line_search_iterations=params['max_line_search_iterations'])

            # tf.print("Function minimum: {0}".format(results.objective_value))
            num_chain_evaluations = [results.num_objective_evaluations]

            if ichain == 0:
                # initialize amplitude and dtime
                amplitude = model.amplitude_ini.numpy()
                amplitude_ini = model.amplitude_ini.numpy()

                dtime = model.dtime_ini.numpy()
                dtime_ini = model.dtime_ini.numpy()

                chain_min = np.zeros(model.nsamples)
                # Check convergence properties
                converged = np.array(results.converged)
                # Check that the argmin is close to the actual value.
                num_evaluations = num_chain_evaluations

                negative_log_likelihood = np.array(results.objective_value)

                if params["train_amplitude"]:
                    amplitude = np.array(results.position)[:, ind_amplitude]
                    amplitude_ini = initial_position[:, ind_amplitude]
                if params["train_dtime"]:
                    dtime = (
                        np.array(results.position)[:, ind_dtime] / params["dtime_norm"]
                    )

                MAPu = np.array(results.position)[:, model.istart_map :]
                MAPu_ini = initial_position[:, model.istart_map :]
                MAPz = np.array(model.flow.bijector.forward(MAPu))
                MAPz_ini = np.array(model.flow.bijector.forward(MAPu_ini))

                if params["train_dtime"]:
                    dtime_ini = initial_position[:, ind_dtime]

            else:
                dm = results.objective_value < negative_log_likelihood

                chain_min[dm] = ichain
                # Check convergence properties
                converged[dm] = np.array(results.converged)[dm]
                # Check that the argmin is close to the actual value.
                num_evaluations += results.num_objective_evaluations

                negative_log_likelihood[dm] = np.array(results.objective_value)[dm]

                #            inv_hessian[dm] = np.array(results.inverse_hessian_estimate)[dm]

                if params["train_amplitude"]:
                    amplitude[dm] = np.array(results.position)[dm, ind_amplitude]
                    amplitude_ini[dm] = initial_position[dm, ind_amplitude]
                if params["train_dtime"]:
                    dtime[dm] = (
                        np.array(results.position)[dm, ind_dtime] / params["dtime_norm"]
                    )
                    dtime_ini[dm] = initial_position[dm, ind_dtime]

                MAPu[dm] = np.array(results.position)[dm, model.istart_map :]
                MAPu_ini[dm] = initial_position[dm, model.istart_map :]

                MAPz[dm] = np.array(model.flow.bijector.forward(MAPu))[dm, :]
                MAPz_ini[dm] = np.array(model.flow.bijector.forward(MAPu_ini))[dm, :]

        if params["verbose"]:
            tf.print(
                f"MAP initialization {ichain} converged. Num function evaluations: {num_chain_evaluations[0]}"
            )

        model.map_results["chain_min"] = tf.Variable(chain_min, dtype=tf.float32)
        model.map_results["converged"] = tf.Variable(converged, dtype=tf.bool)
        model.map_results["num_evaluations"] = tf.Variable(
            num_evaluations, dtype=tf.int32
        )
        model.map_results["num_chain_evaluations"] = tf.Variable(
            num_chain_evaluations, dtype=tf.int32
        )
        model.map_results["negative_log_likelihood"] = tf.Variable(
            negative_log_likelihood, dtype=tf.float32
        )
        model.map_results["amplitude"] = tf.Variable(amplitude, dtype=tf.float32)
        model.map_results["dtime"] = tf.Variable(dtime, dtype=tf.float32)
        model.map_results["MAPu"] = tf.Variable(MAPu, dtype=tf.float32)
        model.map_results["MAPz"] = tf.Variable(MAPz, dtype=tf.float32)
        model.map_results["amplitude_ini"] = tf.Variable(
            amplitude_ini, dtype=tf.float32
        )
        model.map_results["dtime_ini"] = tf.Variable(dtime_ini, dtype=tf.float32)
        model.map_results["MAPu_ini"] = tf.Variable(MAPu_ini, dtype=tf.float32)
        model.map_results["MAPz_ini"] = tf.Variable(MAPz_ini, dtype=tf.float32)
        model.map_results["initial_position"] = tf.Variable(
            initial_position, dtype=tf.float32
        )

        model.save_checkpoint(savepath_i)

    print(f"Min found on chain {chain_min}")

    model.chain_min = tf.convert_to_tensor(model.map_results["chain_min"])
    model.converged = tf.convert_to_tensor(model.map_results["converged"])
    model.num_evaluations = tf.convert_to_tensor(model.map_results["num_evaluations"])
    model.negative_log_likelihood = tf.convert_to_tensor(
        model.map_results["negative_log_likelihood"]
    )
    model.amplitude = tf.convert_to_tensor(model.map_results["amplitude"])
    model.dtime = tf.convert_to_tensor(model.map_results["dtime"])
    model.MAPu = tf.convert_to_tensor(model.map_results["MAPu"])
    model.amplitude_ini = tf.convert_to_tensor(model.map_results["amplitude_ini"])
    model.dtime_ini = tf.convert_to_tensor(model.map_results["dtime_ini"])
    model.MAPu_ini = tf.convert_to_tensor(model.map_results["MAPu_ini"])

    return model


def run_HMC(model, params, savepath_hmc):
    # Initialize the HMC transition kernel.
    # @tf.function(autograph=False)

    if (savepath_hmc / model.ckpt_path).exists():
        print(f"Loading posterior model from {savepath_hmc}")
        model.load_checkpoint(savepath_hmc)

        samples = tf.convert_to_tensor(model.hmc_results["samples"]).numpy()
        step_sizes_final = tf.convert_to_tensor(
            model.hmc_results["step_sizes_final"]
        ).numpy()
        is_accepted = tf.convert_to_tensor(model.hmc_results["is_accepted"]).numpy()
        start = tf.convert_to_tensor(model.hmc_results["start"]).numpy()
        end = tf.convert_to_tensor(model.hmc_results["end"]).numpy()

    else:
        num_warmup_steps = int(params["num_burnin_steps"] * 0.8)

        initial_position = tf.convert_to_tensor(model.MAPu).numpy()
        if params["train_amplitude"] or params["use_amplitude"]:
            # add amplitude as first parameter
            initial_position = np.c_[
                tf.convert_to_tensor(model.amplitude).numpy(), initial_position
            ]
        if params["train_dtime"]:
            initial_position = np.c_[
                tf.convert_to_tensor(model.dtime).numpy()
                * np.array(params["dtime_norm"], dtype=np.float32),
                initial_position,
            ]

        step_sizes = (
            tf.zeros([initial_position.shape[0], initial_position.shape[1]])
            + model.z_latent_std
        )

        progress = tqdm(
            total=params["num_leapfrog_steps"]
            * (params["num_burnin_steps"] + params["num_samples"]),
            leave=True,
        )

        @tf.py_function(Tout=[])
        def update_progress() -> None:
            progress.update()

        @tf.function
        def unnormalized_posterior_log_prob(*args):
            update_progress()
            return model(*args)

        @tf.function
        def trace_fn(_, pkr):
            step_size = pkr.inner_results.accepted_results.step_size
            is_accepted = pkr.inner_results.is_accepted
            return [step_size, is_accepted]

        @tf.function
        def sample_chain(ihmc=True):
            # from https://www.tensorflow.org/probability/examples/TensorFlow_Probability_Case_Study_Covariance_Estimation

            if ihmc:
                # run hmc
                hmc = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=unnormalized_posterior_log_prob,
                    num_leapfrog_steps=params[
                        "num_leapfrog_steps"
                    ],  # to improve convergence
                    step_size=step_sizes,
                )
                #         state_gradients_are_stopped=True)

                kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                    inner_kernel=hmc,
                    num_adaptation_steps=num_warmup_steps,
                    target_accept_prob=params["target_accept_rate"],
                )

                samples, [step_sizes_final, is_accepted] = tfp.mcmc.sample_chain(
                    params["num_samples"],
                    initial_position,
                    kernel=kernel,
                    num_burnin_steps=params["num_burnin_steps"],
                    trace_fn=trace_fn,
                    name="run",
                )

                return samples, step_sizes_final, is_accepted

            # just do random walk
            kernel = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=unnormalized_posterior_log_prob,
            )
            # Run the chain (with burn-in).
            samples, _is_accepted = tfp.mcmc.sample_chain(
                num_results=params["num_samples"],
                num_burnin_steps=params["num_burnin_steps"],
                current_state=initial_position,
                kernel=kernel,
            )
            # parallel_iterations = params['nchains'])

            return samples, np.full(
                (samples[0].shape[0], samples[0].shape[1]), True, dtype=bool
            )

        start = time.time()

        samples, step_sizes_final, is_accepted = sample_chain(params["ihmc"])
        samples, step_sizes_final, is_accepted = (
            samples.numpy(),
            step_sizes_final.numpy(),
            is_accepted.numpy(),
        )
        end = time.time()

        model.hmc_results["samples"] = tf.Variable(samples, dtype=tf.float32)
        model.hmc_results["step_sizes_final"] = tf.Variable(
            step_sizes_final, dtype=tf.float32
        )
        model.hmc_results["is_accepted"] = tf.Variable(is_accepted, dtype=tf.float32)
        model.hmc_results["start"] = tf.Variable(start, dtype=tf.float32)
        model.hmc_results["end"] = tf.Variable(end, dtype=tf.float32)

        model.save_checkpoint(savepath_hmc)

    print(
        "{:.2f} s elapsed for {:d} samples".format(
            end - start, params["num_samples"] + params["num_burnin_steps"]
        )
    )
    # print('Fraction of accepted = ', np.mean(is_accepted), np.mean(is_accepted, axis=0))

    return samples, step_sizes_final, is_accepted, model


def train(PAE, params, train_data, test_data, tstrs=None) -> dict[str, "Any"]:
    if tstrs is None:
        tstrs = ["train", "test"]
    z_latent_std = np.std(train_data["z_latent"][train_data["mask_sn"]], axis=0)
    u_latent_std = np.std(train_data["u_latent"][train_data["mask_sn"]], axis=0)
    z_latent_std[2:] = u_latent_std

    dict_result = {}

    for tstr in tstrs:
        if tstr == "train":
            data_use = train_data

        if tstr == "test":
            data_use = test_data

        nsn = data_use["spectra"].shape[0]

        batch_size = params["batch_size"]
        batch_size = min(batch_size, nsn)
        file_base = os.path.basename(os.path.splitext(params[f"{tstr:s}_data_file"])[0])

        layer_str = "-".join(str(e) for e in params["encode_dims"])
        file_path_out = f"{file_base}_posterior_{params['latent_dim']:02d}Dlatent_layers{layer_str}_{tstr}_{params['posterior_file_tail']}"
        file_path_out = (
            Path(params["PROJECT_DIR"]) / params["OUTPUT_DIR"] / file_path_out
        )

        data_map = {}
        map_results = {}
        hmc_results = {}
        training_hist = {}
        tstart = time.time()
        for batch_start in np.arange(0, nsn, batch_size):
            batch_end = batch_start + batch_size

            print("Posterior analysis of batch ", batch_start // batch_size)

            # Construct new data for batch
            data = {}
            data["spectra"] = data_use["spectra"][batch_start:batch_end]
            data["sigma"] = data_use["sigma"][batch_start:batch_end]
            data["times"] = data_use["times"][batch_start:batch_end]
            data["redshift"] = data_use["redshift"][batch_start:batch_end]
            data["mask"] = data_use["mask"][batch_start:batch_end]
            data["wavelengths"] = data_use["wavelengths"]

            # Get model
            log_posterior = posterior.LogPosterior(
                PAE,
                params,
                data,
                test_data["sigma_ae_time_tbin_cent"],
                test_data["sigma_ae_time"],
            )

            log_posterior.z_latent_std = z_latent_std

            # Parameters to save
            data_map_batch = {}

            data_map_batch["u_latent_ini"] = log_posterior.MAPu_ini
            data_map_batch["amplitude_ini"] = log_posterior.amplitude_ini
            data_map_batch["dtime_ini"] = log_posterior.dtime_ini

            if params["find_MAP"]:
                # Find MAP
                savepath = file_path_out / f"batch_{batch_start}" / "MAP"
                log_posterior = find_MAP(log_posterior, params, savepath)
                map_results[batch_start] = log_posterior.map_results

                # Save desired outputs in dictionary
                data_map_batch["chain_min"] = log_posterior.chain_min
                data_map_batch["converged"] = log_posterior.converged
                data_map_batch["num_evaluations"] = log_posterior.num_evaluations
                data_map_batch["negative_log_likelihood"] = (
                    log_posterior.negative_log_likelihood
                )

                zi = log_posterior.get_z()

                data_map_batch["spectra_map"] = log_posterior.fwd_pass().numpy()
                data_map_batch["u_latent_map"] = log_posterior.MAPu
                data_map_batch["z_latent_map"] = zi.numpy()

                data_map_batch["amplitude_map"] = log_posterior.amplitude

                if params["train_dtime"]:
                    data_map_batch["dtime_map"] = (
                        log_posterior.dtime / params["dtime_norm"]
                    )

                data_map_batch["logp_z_latent_map"] = log_posterior.flow.log_prob(
                    log_posterior.get_z()[:, log_posterior.istart_map :]
                )
                data_map_batch["logp_u_latent_map"] = np.log(
                    1.0
                    / np.sqrt(2 * np.pi)
                    * np.exp(-1.0 / 2 * np.sum(log_posterior.MAPu**2, axis=1))
                )
                data_map_batch["logJ_u_latent_map"] = (
                    log_posterior.flow.bijector.forward_log_det_jacobian(
                        log_posterior.MAPu, event_ndims=1
                    ).numpy()
                )

                # tf.print('evaluation stop={0}:\namplitude: {1}\ndtime {2}'.format(
                #    log_posterior.num_evaluations,
                #    log_posterior.amplitude,
                #    log_posterior.dtime/params['dtime_norm']*50),
                # )

            if params["run_HMC"]:
                # Run HMC
                savepath = file_path_out / f"batch_{batch_start}" / "HMC"
                samples, step_sizes_final, is_accepted, log_posterior = run_HMC(
                    log_posterior, params, savepath
                )
                hmc_results[batch_start] = log_posterior.hmc_results

                z_samples = (
                    log_posterior.flow.bijector.forward(
                        samples[:, :, log_posterior.istart_map :].reshape(
                            -1, log_posterior.latent_dim_u
                        )
                    )
                    .numpy()
                    .reshape(
                        samples.shape[0], samples.shape[1], log_posterior.latent_dim_u
                    )
                )
                ind_amplitude = 0
                ind_dtime = 0
                if params["train_dtime"]:
                    ind_amplitude = 1

                data_map_batch["u_samples"] = samples[:, :, log_posterior.istart_map :]
                if params["train_dtime"]:
                    data_map_batch["dtime_samples"] = (
                        samples[:, :, ind_dtime] / params["dtime_norm"]
                    )

                if params["train_amplitude"]:
                    data_map_batch["amplitude_samples"] = samples[:, :, ind_amplitude]
                    z_samples = np.concatenate(
                        (samples[:, :, 0 : ind_amplitude + 1], z_samples), axis=-1
                    )

                data_map_batch["z_samples"] = z_samples
                data_map_batch["is_accepted"] = is_accepted
                data_map_batch["step_sizes_final"] = step_sizes_final
                # print('final step sizes = ', step_sizes_final.shape, step_sizes_final[-1])

                parameters_mean = np.mean(samples, axis=0)
                parameters_std = np.std(samples, axis=0)
                z_parameters_mean = np.mean(z_samples, axis=0)
                z_parameters_std = np.std(z_samples, axis=0)

                log_posterior.amplitude = parameters_mean[:, ind_amplitude]
                data_map_batch["amplitude_mcmc"] = z_parameters_mean[:, ind_amplitude]
                data_map_batch["amplitude_mcmc_err"] = z_parameters_std[
                    :, ind_amplitude
                ]

                if params["train_dtime"]:
                    data_map_batch["dtime_mcmc"] = (
                        parameters_mean[:, ind_dtime] / params["dtime_norm"]
                    )
                    data_map_batch["dtime_mcmc_err"] = (
                        parameters_std[:, ind_dtime] / params["dtime_norm"]
                    )
                    log_posterior.dtime = (
                        parameters_mean[:, ind_dtime] / params["dtime_norm"]
                    )

                log_posterior.MAPu = parameters_mean[:, log_posterior.istart_map :]
                log_posterior.MAPz = z_parameters_mean[:, log_posterior.istart_map :]

                data_map_batch["u_latent_mcmc"] = parameters_mean[
                    :, log_posterior.istart_map :
                ]
                data_map_batch["u_latent_mcmc_err"] = parameters_std[
                    :, log_posterior.istart_map :
                ]
                data_map_batch["z_latent_mcmc"] = z_parameters_mean
                data_map_batch["z_latent_mcmc_err"] = z_parameters_std

                data_map_batch["spectra_mcmc"] = log_posterior.fwd_pass().numpy()

                # print(log_posterior.MAPu)
                data_map_batch["logp_z_latent_mcmc"] = log_posterior.flow.log_prob(
                    log_posterior.MAPz[:, log_posterior.istart_map :]
                )
                data_map_batch["logp_u_latent_mcmc"] = np.log(
                    1.0
                    / np.sqrt(2 * np.pi)
                    * np.exp(-1.0 / 2 * np.sum(log_posterior.MAPu**2, axis=1))
                )
                data_map_batch["logJ_u_latent_mcmc"] = (
                    log_posterior.flow.bijector.forward_log_det_jacobian(
                        log_posterior.MAPu, event_ndims=1
                    ).numpy()
                )

            if batch_start == 0:
                data_map = data_map_batch.copy()
            else:
                for k in data_map_batch:
                    data_map[k] = np.concatenate((data_map[k], data_map_batch[k]))

        """
        # Get Hessian and covariance matrix at MAP values:
        log_posterior.MAP = tf.convert_to_tensor(log_posterior.MAP)
        log_posterior.amplitude = tf.convert_to_tensor(log_posterior.amplitude)
        log_posterior.dtime = tf.convert_to_tensor(log_posterior.dtime)

        trainable_params, trainable_params_label = setup_trainable_parameters(log_posterior, params)

        map_params = log_posterior.MAP
        if params['train_amplitude']:
            map_params = np.c_[map_params, log_posterior.amplitude]
        if params['train_dtime']:
            map_params = np.c_[map_params, log_posterior.dtime]

        map_params = tf.Variable(tf.convert_to_tensor(map_params))
        hess = get_hessian(log_posterior, map_params, trainable_params)
        """
        tend = time.time()
        print(f"\nTraining took {tend - tstart:.2f} s\n")
        # save to disk
        dicts = [
            data_use,
            data_map,
            {"map_results": map_results},
            {"hmc_results": hmc_results},
        ]
        dict_save = {}
        for d in dicts:
            dict_save.update(dict(d.items()))

        np.savez_compressed(f"{file_path_out}/posterior.npz", **dict_save)
        dict_result[tstr] = dict_save
    return dict_result
