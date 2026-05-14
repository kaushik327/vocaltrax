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

from dataclasses import dataclass
from typing import Any

#######################
# General Settings
#######################

@dataclass
class General:
    # Directory to log results
    log_dir: str
    # Log frequency
    log_every: int
    # Path to target sound
    target: str

    # Sampling rate for the target
    sample_rate: int
    # Frame length
    frame_length: int
    # Hop length
    hop_length: int
    # Upsample glottis
    upsample_glottis: int

    # Number of iterations
    iters: int
    # Learning rate
    lr: float

    # Smoothing blend
    blend: float
    # Smoothing frequency
    smooth_every: int

    # Temporal regularization (replaces Savitzky-Golay with in-loss penalty)
    temporal_regularization: bool
    # Temporal regularization weight (lambda)
    temporal_lambda: float

    # Audio processing parameters
    n_fft: int
    win_length: int
    audio_hop_length: int
    pad: int
    power: float
    n_mels: int
    f_min: float
    f_max: float

    # Enhanced vocal tract features (feature flags)
    add_nose: bool
    add_fricatives: bool
    add_tilt: bool
    add_radiation: bool
    add_wall_loss: bool
    add_postfilter: bool
    add_aspiration: bool
    add_rd: bool

    # Run name for distinguishing experiments
    run_name: str

#######################
# Default Settings
#######################

@dataclass
class Spectrogram:
    # Spectrogram name
    name: str
    fun: Any

@dataclass
class Preloss:
    # Spectrogram name
    name: str
    fun: Any

@dataclass
class Optimizer:
    # Spectrogram name
    name: str
    fun: Any

######################
# System Settings
######################

@dataclass
class System:
    # Random seed for reproducibility
    seed: int
    # Device to run process on. Options: ["cpu", "cuda"]
    device: str

######################
# The Config
######################

@dataclass
class Config:
    # Spectrogram settings
    spectrogram: Spectrogram
    # Preloss settings
    preloss: Preloss
    # Optimizer settings
    optimizer: Optimizer
    # General settings
    general: General
    # System settings
    system: System
