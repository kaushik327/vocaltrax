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
from flax import linen as nn
from tongue import Tongue
from constriction import ThroatConstriction, LipConstriction
from glottis import glottis_make_waveform
from utils.misc import unnormalize_params, upsample_frames


class PhysicalTract(nn.Module):
    num_frames: int
    min_diam: float = 0.2
    max_diam: float = 1.5
    num_segments: int = 44
    base_diams: chex.Array = jnp.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.1,
                                        1.1, 1.1, 1.1, 1.1, 1.5, 1.5, 1.5, 1.5,
                                        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                        1.5, 1.5, 1.5, 1.5])

    def setup(self):
        tongue = Tongue()
        throatconstriction = ThroatConstriction()
        lipconstriction = LipConstriction()

        self.tongue_init_fn = lambda rng, diams: jax.vmap(
            tongue.init,
            in_axes=0
        )(jax.random.split(
                rng,
                self.num_frames
            ),
          diams
          )
        self.throatconstriction_init_fn = lambda rng, diams: jax.vmap(
            throatconstriction.init,
            in_axes=0
        )(jax.random.split(
            rng,
            self.num_frames
        ),
          diams
          )
        self.lipconstriction_init_fn = lambda rng, diams: jax.vmap(
            lipconstriction.init,
            in_axes=0
        )(jax.random.split(
            rng,
            self.num_frames
        ),
          diams
          )

        self.tongue_params = self.param(
            "tongue",
            self.tongue_init_fn,
            jnp.tile(self.base_diams, (self.num_frames, 1))
        )
        self.throatconstriction_params = self.param(
            "throatconstriction",
            self.throatconstriction_init_fn,
            jnp.tile(self.base_diams, (self.num_frames, 1))
        )
        self.lipconstriction_params = self.param(
            "lipconstriction",
            self.lipconstriction_init_fn,
            jnp.tile(self.base_diams, (self.num_frames, 1))
        )

        self.apply_tongue = jax.vmap(tongue.apply)
        self.apply_throatconstriction = jax.vmap(throatconstriction.apply)
        self.apply_lipconstriction = jax.vmap(lipconstriction.apply)

    def __call__(self):
        base_diams = jnp.tile(
            unnormalize_params(self.base_diams, self.min_diam, self.max_diam),
            (self.num_frames, 1)
        )
        tongue_output = self.apply_tongue(self.tongue_params, base_diams)
        diams = self.apply_throatconstriction(self.throatconstriction_params, tongue_output)
        diams = self.apply_lipconstriction(self.lipconstriction_params, diams)
        return diams


def apply_biquad(x: chex.Array, coeffs: chex.Array) -> chex.Array:
    """Apply a biquad (2nd-order IIR) filter to signal x.
    coeffs: [b0, b1, b2, a1, a2] — a0 is implicitly 1.0.
    """
    b0, b1, b2, a1, a2 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]

    def step(carry, x_n):
        x1, x2, y1, y2 = carry
        y_n = b0 * x_n + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        return (x_n, x1, y_n, y1), y_n

    init = (0.0, 0.0, 0.0, 0.0)
    _, y = jax.lax.scan(step, init, x)
    return y


class VocalTract(nn.Module):
    num_frames: int
    f0s: chex.Array
    upsample_factor: int
    frame_len: int
    sample_rate: int

    min_tenseness: float = 0.1
    max_tenseness: float = 1.0
    min_reflection: float = -0.9
    max_reflection: float = 0.9

    glottal_reflection: float = 0.75
    lip_reflection: float = -0.85

    # Nasal tract parameters
    nasal_segments: int = 28
    velum_idx: int = 10

    # Feature flags
    add_nose: bool = False
    add_fricatives: bool = False
    add_tilt: bool = False
    add_radiation: bool = False
    add_wall_loss: bool = False
    add_postfilter: bool = False
    add_aspiration: bool = False
    add_rd: bool = False

    def setup(self):
        self.key = self.make_rng("key")

        physical = PhysicalTract(
            num_frames=self.num_frames
        )
        self.physical_apply = physical.apply
        self.physical_params = self.param(
            "physical",
            lambda rng: physical.init(rng)
            )
        self.tenses = self.param(
            "tenses",
            lambda rng: jnp.ones((self.num_frames, 1))
        )

        # Velum aperture: 0 = closed (oral only), 1 = fully open (nasal coupling)
        if self.add_nose:
            self.velum_aperture = self.param(
                "velum_aperture",
                lambda rng: jnp.zeros((self.num_frames, 1))
            )

        # Glottal spectral tilt: controls 1st-order lowpass cutoff on glottal source
        # 0 = no filtering, 1 = strong lowpass (breathy)
        if self.add_tilt:
            self.glottal_tilt = self.param(
                "glottal_tilt",
                lambda rng: jnp.ones((self.num_frames, 1)) * 0.5
            )

        # Fricative intensity: scales turbulence noise at constrictions
        if self.add_fricatives:
            self.fricative_intensity = self.param(
                "fricative_intensity",
                lambda rng: jnp.zeros((self.num_frames, 1))
            )

        # Learnable Rd parameter: controls voice quality (0.5=pressed, 2.7=breathy)
        # Init at 0.5 (maps to Rd ~1.6 via unnormalize)
        if self.add_rd:
            self.rd_params = self.param(
                "rd_params",
                lambda rng: jnp.ones((self.num_frames, 1)) * 0.5
            )

        # Learnable per-frame aspiration amplitude
        # Init at 0.2 (similar to default aspiration level)
        if self.add_aspiration:
            self.aspiration_amp = self.param(
                "aspiration_amp",
                lambda rng: jnp.ones((self.num_frames, 1)) * 0.2
            )


    def __call__(self):
        diams = self.physical_apply(self.physical_params)
        tenses = unnormalize_params(
            self.tenses,
            self.min_tenseness,
            self.max_tenseness
        )

        f0s = self.f0s

        tenses = upsample_frames(tenses, self.upsample_factor)
        f0s = upsample_frames(f0s, self.upsample_factor)

        # Prepare optional glottal source params
        rd = None
        if self.add_rd:
            rd = unnormalize_params(
                jnp.clip(self.rd_params, 0, 1), 0.5, 2.7
            )
            rd = upsample_frames(rd, self.upsample_factor)

        asp = None
        if self.add_aspiration:
            asp = jnp.clip(self.aspiration_amp, 0, 1)
            asp = upsample_frames(asp, self.upsample_factor)

        waveform = glottis_make_waveform(
            tenses,
            f0s,
            jnp.zeros(self.frame_len//self.upsample_factor),
            self.sample_rate,
            self.key,
            rd_params=rd,
            aspiration_amp=asp,
        )

        # Prepare optional parameters
        velum = None
        if self.add_nose:
            velum = unnormalize_params(
                jnp.clip(self.velum_aperture, 0, 1), 0.0, 0.5
            )
            velum = upsample_frames(velum, self.upsample_factor)

        glottal_tilt = None
        if self.add_tilt:
            glottal_tilt = jnp.clip(self.glottal_tilt, 0, 1)
            glottal_tilt = upsample_frames(glottal_tilt, self.upsample_factor)

        fricative = None
        if self.add_fricatives:
            fricative = jnp.clip(self.fricative_intensity, 0, 1)
            fricative = upsample_frames(fricative, self.upsample_factor)

        out = process_diams(
            waveform,
            diams,
            self.glottal_reflection,
            self.lip_reflection,
            frame_size=self.frame_len,
            velum_aperture=velum,
            velum_idx=self.velum_idx,
            nasal_segments=self.nasal_segments,
            glottal_tilt=glottal_tilt,
            fricative_intensity=fricative,
            key=self.key,
            add_nose=self.add_nose,
            add_fricatives=self.add_fricatives,
            add_tilt=self.add_tilt,
            add_radiation=self.add_radiation,
            add_wall_loss=self.add_wall_loss,
        )

        return out


def process_diams(
    input: chex.Array,
    diameters: chex.Array,
    glottal_reflection: float = 0.75,
    lip_reflection: float = -0.85,
    frame_size: int = 1024,
    velum_aperture: chex.Array = None,
    velum_idx: int = 10,
    nasal_segments: int = 28,
    glottal_tilt: chex.Array = None,
    fricative_intensity: chex.Array = None,
    key: jax.random.PRNGKey = None,
    add_nose: bool = False,
    add_fricatives: bool = False,
    add_tilt: bool = False,
    add_radiation: bool = False,
    add_wall_loss: bool = False,
) -> chex.Array:
    """
    Vocal tract waveguide simulation with optional enhanced features:
    - add_nose: Nasal tract coupling at velum
    - add_fricatives: Turbulence noise injection at constrictions
    - add_tilt: Glottal spectral tilt (1st-order lowpass)
    - add_radiation: Radiation impedance (1st-order highpass at lips)
    - add_wall_loss: Frequency-dependent wall losses
    
    When all flags are False, behaves identically to the original implementation.
    """
    A = diameters**2
    reflections = (A[:, :-1] - A[:, 1:]) / (A[:, :-1] + A[:, 1:] + 1e-12)

    n_frames, size = jnp.shape(diameters)
    n = (n_frames - 1) * frame_size
    reflections = upsample_frames(reflections, frame_size).reshape((n, size - 1))
    input = jnp.pad(input, (0, n - input.size), mode='constant', constant_values=0)

    # Upsample per-sample parameters
    if velum_aperture is not None:
        velum_aperture = upsample_frames(velum_aperture, frame_size).ravel()[:n]
    else:
        velum_aperture = jnp.zeros(n)

    if glottal_tilt is not None:
        glottal_tilt = upsample_frames(glottal_tilt, frame_size).ravel()[:n]
    else:
        glottal_tilt = jnp.zeros(n)

    if fricative_intensity is not None:
        fricative_intensity = upsample_frames(fricative_intensity, frame_size).ravel()[:n]
    else:
        fricative_intensity = jnp.zeros(n)

    # Frequency-dependent wall losses (or uniform if disabled)
    if add_wall_loss:
        segment_loss = 0.999 - 0.002 * jnp.linspace(0, 1, size)
    else:
        segment_loss = jnp.ones(size) * 0.999

    # Nasal tract setup (only if enabled)
    if add_nose:
        nasal_diams = jnp.concatenate([
            jnp.ones(nasal_segments - 3) * 1.2,
            jnp.array([0.8, 0.5, 0.3])
        ])
        nasal_A = nasal_diams**2
        nasal_reflections = (nasal_A[:-1] - nasal_A[1:]) / (nasal_A[:-1] + nasal_A[1:] + 1e-12)
        if add_wall_loss:
            nasal_loss = 0.997 - 0.002 * jnp.linspace(0, 1, nasal_segments)
        else:
            nasal_loss = jnp.ones(nasal_segments) * 0.997
    else:
        nasal_reflections = jnp.zeros(nasal_segments - 1)
        nasal_loss = jnp.ones(nasal_segments) * 0.997

    # Initialize waveguide states
    R = jnp.zeros(size)
    L = jnp.zeros(size)
    R_n = jnp.zeros(nasal_segments)
    L_n = jnp.zeros(nasal_segments)

    rad_state = 0.0
    tilt_state = 0.0
    rad_alpha = 0.93

    # Generate noise for fricatives
    if add_fricatives and key is not None:
        noise = jax.random.normal(key, (n,)) * 0.1
    else:
        noise = jnp.zeros(n)

    def simulation_step(carry, x):
        R, L, R_n, L_n, rad_state, tilt_state = carry
        reflection, input_sample, velum_t, tilt_t, fric_t, noise_t = x

        # 1. Glottal spectral tilt (if enabled)
        if add_tilt:
            tilt_coef = 0.3 + 0.6 * tilt_t
            tilt_out = tilt_coef * tilt_state + (1 - tilt_coef) * input_sample
            tilt_state_new = tilt_out
            input_filtered = input_sample * (1 - tilt_t) + tilt_out * tilt_t
        else:
            tilt_state_new = tilt_state
            input_filtered = input_sample

        # 2. Fricative noise (if enabled)
        # Only inject noise when there's a tight constriction (reflection > threshold)
        if add_fricatives:
            constriction_threshold = 0.5
            max_reflection = jnp.max(jnp.abs(reflection))
            # Scale by how much constriction exceeds threshold (0 if below)
            constriction_strength = jnp.maximum(0.0, max_reflection - constriction_threshold)
            fric_noise = noise_t * fric_t * constriction_strength * 4.0
        else:
            fric_noise = 0.0

        # Oral tract junctions
        junction_outR = jnp.zeros(size + 1)
        junction_outL = jnp.zeros(size + 1)

        junction_outR = junction_outR.at[0].set(
            L[0] * glottal_reflection + input_filtered
        )

        # 3. Nasal coupling (if enabled)
        if add_nose:
            oral_at_velum = R[velum_idx - 1]
            nasal_coupling = velum_t * oral_at_velum * 0.5
        else:
            nasal_coupling = 0.0

        # Standard oral reflections
        w = reflection * (R[:-1] + L[1:])
        junction_outR = junction_outR.at[1:-1].set(R[:-1] - w)
        junction_outL = junction_outL.at[1:-1].set(L[1:] + w)

        # Inject fricative noise at constriction point
        if add_fricatives:
            max_refl_idx = jnp.argmax(jnp.abs(reflection))
            junction_outR = junction_outR.at[max_refl_idx + 1].add(fric_noise)

        # 4. Radiation impedance (if enabled) or simple reflection
        lip_in = R[size - 1]
        if add_radiation:
            rad_out = lip_in - rad_state
            rad_state_new = rad_alpha * rad_state + (1 - rad_alpha) * lip_in
            junction_outL = junction_outL.at[size].set(-rad_out * 0.85)
        else:
            rad_out = lip_in
            rad_state_new = rad_state
            junction_outL = junction_outL.at[size].set(lip_in * lip_reflection)

        # 5. Wall losses (frequency-dependent or uniform, already set in segment_loss)
        R_new = junction_outR[:size] * segment_loss
        L_new = junction_outL[1:size + 1] * segment_loss

        # Nasal tract simulation (if enabled)
        if add_nose:
            nasal_junction_outR = jnp.zeros(nasal_segments + 1)
            nasal_junction_outL = jnp.zeros(nasal_segments + 1)

            nasal_junction_outR = nasal_junction_outR.at[0].set(
                L_n[0] * 0.75 + nasal_coupling
            )
            nasal_junction_outL = nasal_junction_outL.at[nasal_segments].set(
                R_n[nasal_segments - 1] * (-0.7)
            )

            w_n = nasal_reflections * (R_n[:-1] + L_n[1:])
            nasal_junction_outR = nasal_junction_outR.at[1:-1].set(R_n[:-1] - w_n)
            nasal_junction_outL = nasal_junction_outL.at[1:-1].set(L_n[1:] + w_n)

            R_n_new = nasal_junction_outR[:nasal_segments] * nasal_loss
            L_n_new = nasal_junction_outL[1:nasal_segments + 1] * nasal_loss

            nasal_out = R_n[nasal_segments - 1] * velum_t
            output1 = rad_out + nasal_out * 0.5
        else:
            R_n_new = R_n
            L_n_new = L_n
            output1 = rad_out

        # Second half-sample update
        junction_outR2 = jnp.zeros(size + 1)
        junction_outL2 = jnp.zeros(size + 1)
        junction_outR2 = junction_outR2.at[0].set(L_new[0] * glottal_reflection)

        w2 = reflection * (R_new[:-1] + L_new[1:])
        junction_outR2 = junction_outR2.at[1:-1].set(R_new[:-1] - w2)
        junction_outL2 = junction_outL2.at[1:-1].set(L_new[1:] + w2)

        lip_in2 = R_new[size - 1]
        if add_radiation:
            rad_out2 = lip_in2 - rad_state_new
            rad_state_new2 = rad_alpha * rad_state_new + (1 - rad_alpha) * lip_in2
            junction_outL2 = junction_outL2.at[size].set(-rad_out2 * 0.85)
        else:
            rad_out2 = lip_in2
            rad_state_new2 = rad_state_new
            junction_outL2 = junction_outL2.at[size].set(lip_in2 * lip_reflection)

        R_new2 = junction_outR2[:size] * segment_loss
        L_new2 = junction_outL2[1:size + 1] * segment_loss

        if add_nose:
            output2 = rad_out2 + R_n_new[nasal_segments - 1] * velum_t * 0.5
        else:
            output2 = rad_out2

        return (R_new2, L_new2, R_n_new, L_n_new, rad_state_new2, tilt_state_new), jnp.array([output1, output2])

    # Run simulation
    init_carry = (R, L, R_n, L_n, rad_state, tilt_state)
    scan_inputs = (reflections, input, velum_aperture, glottal_tilt, fricative_intensity, noise)
    _, out = jax.lax.scan(simulation_step, init_carry, scan_inputs)

    out = jnp.append(0, jnp.ravel(out))
    out = out[1:] + out[:-1]

    return out[1::2] * 0.25
