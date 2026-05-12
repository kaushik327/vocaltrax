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
import math
from audax.core import functional
from functools import partial
from scipy.signal import resample_poly


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    gcd = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // gcd
    down = int(orig_sr) // gcd
    return resample_poly(audio, up, down).astype(audio.dtype)

def _make_spec_func(
        n_fft,
        win_length,
        hop_length,
        pad,
        power
):
    window = jnp.hanning(win_length)
    return partial(
        functional.spectrogram,
        pad=pad,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=False,
        center=True,
        onesided=True,
    )

def _make_mel_func(
        n_fft,
        n_mels,
        sample_rate,
        f_min,
        f_max
):
    fb = functional.melscale_fbanks(
        n_freqs=(n_fft // 2) + 1, n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max
    )
    return partial(functional.apply_melscale, melscale_filterbank=fb)


def spec(
        audio_input: chex.Array,
        n_fft,
        win_length,
        hop_length,
        pad,
        power
):
    spec_func = _make_spec_func(n_fft, win_length, hop_length, pad, power)
    num_samples = len(audio_input)
    num_chunks = int(np.ceil(num_samples / 100000))
    audio_input = jnp.pad(audio_input, (0, num_chunks * 100000 - num_samples), mode='constant')
    audio_input = jnp.reshape(audio_input, (num_chunks, 100000))
    return spec_func(audio_input)


def melspec(
        audio_input:
        chex.Array,
        n_fft,
        win_length,
        hop_length,
        pad,
        power,
        n_mels,
        sample_rate,
        f_min,
        f_max
):
    spec_func = _make_spec_func(n_fft, win_length, hop_length, pad, power)
    mel_spec_func = _make_mel_func(n_fft, n_mels, sample_rate, f_min, f_max)
    num_samples = len(audio_input)
    num_chunks = int(np.ceil(num_samples / 100000))
    audio_input = jnp.pad(audio_input, (0, num_chunks * 100000 - num_samples), mode='constant')
    audio_input = jnp.reshape(audio_input, (num_chunks, 100000))
    return mel_spec_func(spec_func(audio_input))


def multiscalemel(
        audio_input: chex.Array,
        n_fft,
        pad,
        power,
        n_mels,
        sample_rate,
        f_min,
        f_max
):
    mel_spec_func = _make_mel_func(n_fft, n_mels, sample_rate, f_min, f_max)
    return jnp.concatenate(
        [
            mel_spec_func(
                partial(
                    functional.spectrogram,
                    pad=pad,
                    window=jnp.hanning(size),
                    n_fft=n_fft,
                    hop_length=size // 2,
                    win_length=size,
                    power=power,
                    normalized=False,
                    center=True,
                    onesided=True,
                )(audio_input)
            )
            for size in [512, 1024, 2048]
        ],
        axis=1,
    )


def combined(
        audio_input: chex.Array,
        n_fft,
        win_length,
        hop_length,
        pad,
        power,
        n_mels,
        sample_rate,
        f_min,
        f_max
):
    spec_func = _make_spec_func(n_fft, win_length, hop_length, pad, power)
    mel_spec_func = _make_mel_func(n_fft, n_mels, sample_rate, f_min, f_max)
    return jnp.concatenate(
        [spec_func(audio_input), mel_spec_func(spec_func(audio_input))], axis=2
    )
