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
from utils.misc import upsample_frames

def glottis_make_waveform(
    tenseness: chex.Array,
    freqs: chex.Array,
    frame_len: chex.Array,
    sample_rate: int,
    key: jax.random.PRNGKey,
    rd_params: chex.Array = None,
    aspiration_amp: chex.Array = None,
) -> chex.Array:
    """
    Args:
        tenseness: [n_frames x 1]
        freqs: [n_frames x 1]
        frame_len: [frame_len x 1]
        rd_params: optional [n_frames x 1], learnable Rd values (0.5-2.7)
        aspiration_amp: optional [n_frames x 1], learnable aspiration amplitude
    """
    T = 1 / sample_rate
    n_frames = tenseness.shape[0]
    if n_frames > 1:
        n_frames -= 1
        frame_len = jnp.shape(frame_len)[0]
        n = frame_len * n_frames
        wav_len = 1 / jnp.ravel(upsample_frames(freqs, frame_len))
        t = jnp.arange(n) * T

        def process_accumulate(accumulate, new):
            ti, wav_leni = new
            condition = ti + accumulate > wav_leni
            accumulate = jnp.where(condition, accumulate - wav_leni, accumulate)
            return accumulate, accumulate

        _, accumulate = jax.lax.scan(process_accumulate, 0.0, (t, wav_len))
        t += accumulate

        t /= wav_len
    else:
        n = frame_len
        wav_len = 1 / freqs[0]
        t = jnp.mod(jnp.arange(n) * T, wav_len) / wav_len

    # Prepare per-frame Rd and aspiration arrays
    if rd_params is not None:
        rd_per_frame = rd_params[:-1].ravel()
    else:
        rd_per_frame = None

    if aspiration_amp is not None:
        asp_per_frame = aspiration_amp[:-1].ravel()
    else:
        asp_per_frame = None

    keys = jax.random.split(key, n_frames)
    def one_frame(carry, input):
        tensenessi, t_slice, key, rd_i, asp_i = input
        # Use learnable Rd if provided, otherwise derive from tenseness
        lf_params = setup_lf(tensenessi, rd_override=rd_i)

        greaterIdx = t_slice > lf_params["Te"]
        result = jnp.where(
            greaterIdx,
            (-jnp.exp(-lf_params["epsilon"] * (t_slice - lf_params["Te"])) + lf_params["shift"])
            / lf_params["delta"],
            lf_params["EO"]
            * jnp.exp(lf_params["alpha"] * t_slice)
            * jnp.sin(lf_params["omega"] * t_slice),
        )
        result *= (tensenessi**0.25)

        # White Noise Addition (with optional learnable amplitude)
        # Tenseness envelope gates aspiration: breathy segments get more noise
        tense_gate = (1 - jnp.sqrt(tensenessi))
        # When learnable asp_i is active, it scales the gated noise; otherwise use default
        asp_scale = jnp.where(asp_i >= 0, asp_i * tense_gate, tense_gate * 0.2)
        aspiration = asp_scale * (jax.random.uniform(key, (frame_len,)) - 0.5) * 0.2
        result += aspiration
        return 0.0, result

    # Build scan inputs — use sentinel -1.0 when feature is not active
    rd_arr = rd_per_frame if rd_per_frame is not None else jnp.full(n_frames, -1.0)
    asp_arr = asp_per_frame if asp_per_frame is not None else jnp.full(n_frames, -1.0)

    _, result = jax.lax.scan(
        one_frame, 0.0, (tenseness[:-1], t.reshape(n_frames, frame_len), keys, rd_arr, asp_arr)
    )
    return jnp.ravel(result)

def setup_lf(tenseness, rd_override=None):
    """
    Returns parameters of Liljencrants-Fant (LF) model of GFD waveform based on tenseness.
    https://www.tara.tcd.ie/bitstream/handle/2262/92586/Gobl%20-%20Reshaping%20the%20transformed%20LF%20model%20-%20Interspeech2017.pdf;jsessionid=766CC7D94CED11EDA65F1098918A18D1?sequence=1

    If rd_override >= 0, use it directly as Rd. Otherwise derive from tenseness.
    """
    Rd_from_tense = 3 * (1 - tenseness)
    Rd = jnp.where(rd_override >= 0, rd_override, Rd_from_tense)
    Rd = jnp.clip(Rd, 0.5, 2.7)

    Ra = -0.01 + 0.048 * Rd
    Rk = 0.224 + 0.118 * Rd
    Rg = (Rk / 4) * (0.5 + 1.2 * Rk) / (0.11 * Rd - Ra * (0.5 + 1.2 * Rk))

    Ta = Ra
    Tp = 1 / (2 * Rg)
    Te = Tp + Tp * Rk

    epsilon = 1 / Ta
    shift = jnp.exp(-epsilon * (1 - Te))
    delta = 1 - shift

    rhs_integral = (1 / epsilon) * (shift - 1) + (1 - Te) * shift
    rhs_integral /= delta

    lower_integral = -(Te - Tp) / 2 + rhs_integral
    upper_integral = -lower_integral

    omega = jnp.pi / Tp
    s = jnp.sin(omega * Te)
    y = -jnp.pi * s * upper_integral / (Tp * 2)
    z = jnp.log(y)
    alpha = z / (Tp / 2 - Te)
    EO = -1 / (s * jnp.exp(alpha * Te))

    return {
        "alpha": alpha,
        "EO": EO,
        "epsilon": epsilon,
        "shift": shift,
        "delta": delta,
        "Te": Te,
        "omega": omega,
    }
