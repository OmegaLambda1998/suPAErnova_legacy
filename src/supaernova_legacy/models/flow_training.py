#!/usr/bin/env python
"""This code constructs and trains the normalizing flow model,
based on the parameters specified in the configuration file, config/train.yaml.

The Autoencoder architecture is specified in models/autoencoder.py,
and the loss terms are specified in models/losses.py.
"""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_probability as tfp
from tensorflow_probability import (
    bijectors as tfb,
    distributions as tfd,
)

from . import (
    flows,
)

# print("tensorflow version: ", tf.__version__)
# print("devices: ", tf.config.list_physical_devices("GPU"))
# print("TFK Version", ks.__version__)
# print("TFP Version", tfp.__version__)


def train_flow(train_data, test_data, params):
    """Train a simple MAF model for density estimation.
    Can definitely be improved/should be later,
    as the flow does not always train well in high dimensions.
    """
    optimizer = ks.optimizers.Adam(params["lr_flow"])

    # Don't use time shift or amplitude in normalizing flow
    # Amplitude represents uncorrelated shift from peculiar velocity and/or gray instrumental effects
    # And this is the parameter we want to fit to get "cosmological distances", thus we don't want a prior on it
    istart = 2 if params["use_extrinsic_params"] else 3

    z_latent = tf.convert_to_tensor(
        train_data["z_latent"][:, istart:], dtype=tf.float32
    )

    print("Size of training data = ", z_latent.shape)
    layers_str = "-".join(str(e) for e in params["encode_dims"])
    checkpoint_filepath = (
        f"{params['NFLOW_MODEL_DIR']}flow_kfold{params['kfold']}_{params['latent_dim']:02d}Dlatent_"
        f"layers{layers_str}_nlayers{params['nlayers']:02d}_{params['out_file_tail']}/"
    )

    cp_callback = ks.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            params["PROJECT_DIR"],
            checkpoint_filepath,
        ),
        save_weights_only=True,
        verbose=params["verbose"],
        save_freq=min(params["checkpoint_flow_every"], params["epochs_flow"]),
    )

    earlystopping_callback = ks.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=params["patience"],
    )

    NFmodel, flow = flows.normalizing_flow(params, optimizer=optimizer)

    # Dummy run
    # print(z_latent.shape)
    dummy = NFmodel(z_latent, training=False)
    # dummy_loss = NFmodel.loss(dummy, dummy)
    # us = flow.bijector.inverse(z_latent)
    # print(z_latent, us, dummy, dummy_loss)

    NFmodel.fit(
        x=z_latent,
        y=tf.zeros_like(z_latent, dtype=tf.float32),
        validation_split=params["val_frac_flow"],
        batch_size=params["batch_size"],
        epochs=params["epochs_flow"],
        steps_per_epoch=z_latent.shape[0] // params["batch_size"],
        shuffle=True,
        verbose=params["verbose"],
        callbacks=[cp_callback, earlystopping_callback],
    )

    NFmodel.trainable = False
    tf.train.Checkpoint(
        NFmodel,
    ).save(checkpoint_filepath)

    print("Done training flow!")

    return NFmodel, flow
