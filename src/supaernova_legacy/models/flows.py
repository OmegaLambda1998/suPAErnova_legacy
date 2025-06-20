import os
import random as rn

import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow_probability import (
    bijectors as tfb,
    distributions as tfd,
)


def normalizing_flow(params, optimizer=None):
    """event_dim: dimensions of input data."""
    # set random seeds
    os.environ["PYTHONHASHSEED"] = str(params["seed"])
    tf.random.set_seed(params["seed"])
    np.random.seed(params["seed"])
    rn.seed(params["seed"])

    train_phase = True
    if optimizer is None:
        optimizer = ks.optimizers.Adam(1e-3)

    # Don't use time shift or amplitude in normalizing flow
    # Amplitude represents uncorrelated shift from peculiar velocity and/or gray instrumental effects
    # And this is the paramater we want to fit to get "cosmological distances"
    u_latent_dim = params["latent_dim"]
    if params["use_extrinsic_params"]:
        u_latent_dim += 1  # plus one to include color term

    indices = np.roll(np.arange(u_latent_dim), 1)
    permutations = [indices for ii in range(params["nlayers"])]

    bijectors = []
    if params["batchnorm"]:
        bijectors.append(
            tfb.BatchNormalization(
                training=train_phase,
                name="batch_normalization",
            )
        )

    for i in range(params["nlayers"]):
        bijectors.append(
            tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                    params=2,
                    hidden_units=[params["nunit"], params["nunit"]],
                    activation="relu",
                    use_bias=True,
                )
            )
        )
        if params["batchnorm"]:
            bijectors.append(
                tfb.BatchNormalization(
                    training=train_phase,
                    name="batch_normalization",
                )
            )

        bijectors.append(tfb.Permute(permutation=permutations[i]))

    # Construct flow model
    flow = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(
            loc=tf.zeros(u_latent_dim),
            scale_diag=tf.ones(u_latent_dim),
        ),
        bijector=tfb.Chain(list(reversed(bijectors[:-1]))),
    )

    z_ = ks.layers.Input(
        shape=(u_latent_dim,),
        dtype=tf.float32,
    )

    log_prob_ = flow.log_prob(z_)

    model = ks.Model(
        inputs=z_,
        outputs=log_prob_,
    )

    model.compile(
        optimizer=optimizer,
        loss=lambda _, log_prob: -log_prob,
        # run_eagerly=True,
    )

    return model, flow
