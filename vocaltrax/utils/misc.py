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

import jax
import jax.numpy as jnp
import chex
import numpy as np

def frameaudio(
    x: chex.Array,
    frame_length: int,
    hop_length: int,
    axis: int = -1
) -> chex.Array:
    starts = jnp.arange(0, len(x) - frame_length + 1, hop_length)
    xw = jax.vmap(lambda start: jax.lax.dynamic_slice(x, (start,), (frame_length,)))(
        starts
    )
    if axis == -1:
        return jnp.transpose(xw)
    return xw

def upsample_frames(frames, upsample_factor):
    n_frames, n_dims = frames.shape
    if n_frames == 1:
        samples = jnp.repeat(frames, upsample_factor, axis=0)
        return samples
    interp = jnp.array([jnp.arange(upsample_factor) / upsample_factor])

    def one_frame(carry, x):
        curr = jnp.array([x])
        return curr, jnp.transpose(
            jnp.matmul(jnp.transpose(carry), (1 - interp))
            + jnp.matmul(jnp.transpose(curr), interp)
        )

    _, stacked_samples = jax.lax.scan(one_frame, jnp.array([frames[0]]), frames[1:])
    return jnp.reshape(stacked_samples, (upsample_factor * (n_frames - 1), n_dims))

def unnormalize_params(normalized_params, minimum, maximum):
    return minimum + normalized_params * (maximum - minimum)

def preloss_norm(spec):
    return spec

def preloss_log(spec):
    return jnp.log10(jnp.abs(spec) + 1e-6)

def preloss_loudness(spec, n_mels, sample_rate, f_min, f_max):
    from utils.audio import perceptual_weight_vector
    w = perceptual_weight_vector(n_mels, sample_rate, f_min, f_max)
    return jnp.log10(jnp.abs(spec) + 1e-6) * w

def mse(pred, target):
    return jnp.mean(jnp.square((pred - target)))

def jax_to_numpy(obj):
    if isinstance(obj, jnp.ndarray):
        return np.array(obj).tolist()
    elif isinstance(obj, dict):
        return {k: jax_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [jax_to_numpy(v) for v in obj]
    else:
        return obj
