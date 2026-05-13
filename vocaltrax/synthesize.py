# Copyright (c) 2025
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>
# Luke Mo <lukemo@mit.edu>
# Quinn Langford <langford@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import hydra
import jax
import jax.numpy as jnp
import optax
import soundfile
import json
import crepe
import numpy as np
import scipy.signal
from omegaconf import OmegaConf
from config import Config
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tqdm import tqdm
from tract import VocalTract
from utils.random import PRNGKey
from utils.hydra import print_config
from utils.misc import frameaudio, mse, jax_to_numpy

config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Config) -> None:
    # Print config
    OmegaConf.resolve(cfg)
    print_config(cfg)

    ##############################################
    # Seeding
    ##############################################

    PRNG_key = PRNGKey(cfg.system.seed)

    ##############################################
    # Log dir
    ##############################################

    current_datetime = datetime.now().strftime("%a-%b-%d-%Y_%I-%M%p")
    run_suffix = f"_{cfg.general.run_name}" if cfg.general.run_name else ""
    log_dir = os.path.join(
        cfg.general.log_dir,
        os.path.splitext(os.path.basename(cfg.general.target))[0], # Target name
        cfg.optimizer.name,
        cfg.spectrogram.name,
        cfg.preloss.name,
        f"{current_datetime}{run_suffix}"
    )
    os.makedirs(log_dir, exist_ok=True)

    ##############################################
    # Load audio file
    ##############################################

    target, sr = soundfile.read(cfg.general.target)
    if len(target.shape) == 2: target = jnp.mean(target, axis=1)
    target = jnp.pad(
        target,
        (cfg.general.frame_length - (len(target) % cfg.general.frame_length)) % cfg.general.frame_length,
        mode='constant',
        constant_values=0
    )

    ##############################################
    # Divide audio in multiple frames
    ##############################################

    frames = frameaudio(
        target,
        frame_length=cfg.general.frame_length,
        hop_length=cfg.general.hop_length,
        axis=0,
    )
    n_frames = len(frames)

    # Get fundamental frequency of each frame using CREPE
    file_hash = sha256(target.tobytes()).hexdigest()[:4]
    cache_path = (
        Path(cfg.general.target).parent
        / ".crepe_cache"
        / f"{Path(cfg.general.target).stem}_sr={sr}_hop={cfg.general.hop_length}_{file_hash}.npz"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached CREPE from {cache_path}")
        freqs = np.load(cache_path)["freqs"]
    else:
        _, freqs, conf, _ = crepe.predict(
            np.array(target),
            sr,
            step_size=np.floor((cfg.general.hop_length / sr) * 1000),
        )
        freqs = freqs * (conf > 0.5)

        # replace zeros with 1.0 to avoid division by zero
        freqs = np.where(freqs > 0, freqs, 1.0)

        np.savez(cache_path, freqs=freqs)
        print(f"Saved CREPE cache to {cache_path}")

    freqs = freqs[:n_frames]

    ##############################################
    # Instantiate components
    ##############################################

    tract = VocalTract(
        num_frames=n_frames,
        upsample_factor=cfg.general.upsample_glottis,
        frame_len=cfg.general.frame_length,
        sample_rate=cfg.general.sample_rate,
        f0s=jnp.array(freqs).reshape(len(freqs), 1),
        add_nose=cfg.general.add_nose,
        add_fricatives=cfg.general.add_fricatives,
        add_tilt=cfg.general.add_tilt,
        add_radiation=cfg.general.add_radiation,
        add_wall_loss=cfg.general.add_wall_loss,
    )
    init_key = PRNG_key.split()
    params = tract.init(init_key)
    tract_apply = jax.jit(tract.apply)

    ##############################################
    # Optimizing loop
    ##############################################

    optimizer = hydra.utils.instantiate(cfg.optimizer.fun, learning_rate=cfg.general.lr)
    opt_state = optimizer.init(params)

    audio = tract_apply(params, rngs={"params": init_key})
    loss_func = jax.jit(mse)

    preloss_func = jax.jit(hydra.utils.instantiate(cfg.preloss.fun, _partial_=True))
    spec_func = jax.jit(hydra.utils.instantiate(cfg.spectrogram.fun, _partial_=True))
    loss_target = preloss_func(spec_func((target[: len(audio)])))

    def loss_fn(params, key):
        # Get prediction and regularizer
        audio = tract_apply(params, rngs={"params": key})
        # Calculate loss
        loss = loss_func(preloss_func(spec_func(audio)), loss_target)

        # Temporal regularization: λ Σ_t ||a^(t+1) - a^(t)||^2
        if cfg.general.temporal_regularization:
            def temporal_penalty(x):
                if (len(x.shape) > 0) and (x.shape[0] == n_frames):
                    return jnp.sum(jnp.diff(x, axis=0) ** 2)
                return 0.0
            reg = jax.tree_util.tree_reduce(
                lambda a, b: a + b,
                jax.tree_util.tree_map(temporal_penalty, params),
            )
            loss = loss + cfg.general.temporal_lambda * reg

        return loss, audio

    # Gradient should use only the 1st element (loss)
    grad_value = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    # Log initial params
    soundfile.write(os.path.join(log_dir, "0.wav"), audio, sr)

    # Optimization loop
    pbar = tqdm(range(cfg.general.iters))
    for i in pbar:
        # Smoothing (skipped when temporal regularization is active)
        if not cfg.general.temporal_regularization and ((i + 1) % cfg.general.smooth_every) == 0:
            params = jax.tree_util.tree_map(
                lambda x : (
                    (
                        jnp.array(
                            scipy.signal.savgol_filter(
                                jax_to_numpy(x), 10, 7, axis=0
                            )
                        ) * cfg.general.blend
                    ) + (x * (1 - cfg.general.blend))
                ) if (len(x.shape) > 0) and (x.shape[0] == n_frames) else x,
                params
            )

        # Calculate gradients
        (loss, audio), grads = grad_value(params, PRNG_key.split())

        # Update optimizer and parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Clip parameters between [0, 1]
        params = jax.tree_util.tree_map(
            lambda x: jnp.clip(x, a_min=0, a_max=1),
            params
        )

        # Log metrics to progress bar
        pbar.set_postfix({"loss": loss.item()})
        if jnp.isnan(loss) or jnp.isinf(loss):
            print(f"\nERROR: Loss became {loss.item()} at step {i}. Aborting.")
            soundfile.write(os.path.join(log_dir, f"{i+1}_diverged.wav"), audio, sr)
            break
        if ((i + 1) % cfg.general.log_every) == 0:
            soundfile.write(
                os.path.join(log_dir, f"{i+1}.wav"), audio, sr
            )

    # Save JAX params
    with open(os.path.join(log_dir, "params.json"), "w") as f:
        f.write(json.dumps(jax_to_numpy(params), indent=4))

if __name__ == "__main__":
    main()
