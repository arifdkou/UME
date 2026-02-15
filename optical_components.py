# optical_components.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PI = np.pi

@dataclass
class LaserSource:
    wavelength_m: float   # meters
    power_w: float        # watts
    m2: float = 1.0

@dataclass
class Fiber:
    na: float
    core_diam_m: float
    # baseline coupling (perfect alignment) e.g. 0.75
    eta0: float = 0.75

@dataclass
class Lens:
    f_m: float
    transmission: float = 0.995  # coating + bulk
    name: str = ""

@dataclass
class BeamState:
    """Gaussian beam with M^2 using q-parameter model."""
    wavelength_m: float
    m2: float
    q: complex            # q-parameter
    power_w: float

def zR_from_w0(w0_m: float, wavelength_m: float, m2: float) -> float:
    # Effective Rayleigh range with M^2
    return PI * w0_m**2 / (wavelength_m * m2)

def q_from_waist(w0_m: float, wavelength_m: float, m2: float, z0_m: float = 0.0) -> complex:
    # q(z) = z + i zR
    zR = zR_from_w0(w0_m, wavelength_m, m2)
    return (z0_m + 1j * zR)

def w_from_q(q: complex, wavelength_m: float, m2: float) -> float:
    # 1/q = 1/R - i (lambda*M^2)/(pi w^2)
    invq = 1.0 / q
    imag = -np.imag(invq)
    # imag = (lambda*M^2)/(pi w^2)
    return np.sqrt((wavelength_m * m2) / (PI * imag))

def propagate_free_space(beam: BeamState, L_m: float) -> BeamState:
    return BeamState(
        wavelength_m=beam.wavelength_m,
        m2=beam.m2,
        q=beam.q + L_m,
        power_w=beam.power_w
    )

def thin_lens(beam: BeamState, f_m: float, lens_transmission: float = 1.0) -> BeamState:
    # ABCD: lens => A=1, B=0, C=-1/f, D=1
    q_in = beam.q
    q_out = 1.0 / (1.0/q_in - 1.0/f_m)
    return BeamState(
        wavelength_m=beam.wavelength_m,
        m2=beam.m2,
        q=q_out,
        power_w=beam.power_w * lens_transmission
    )

def fresnel_transmission(n1: float, n2: float, incidence_rad: float = 0.0) -> float:
    # Simple normal-incidence Fresnel (power) approximation
    # For MVP assume normal incidence
    r = ((n1 - n2) / (n1 + n2))**2
    return 1.0 - r

def gaussian_overlap_eta_lateral(r_m: float, w_m: float) -> float:
    # Power coupling ~ exp(-2 r^2 / w^2)  (common approximation for mode overlap)
    return float(np.exp(-2.0 * (r_m**2) / (w_m**2)))

def coupling_loss_db(r_m: float, w_m: float) -> float:
    # user formula: dB = 4.34 (r/w)^2 (small-offset approx)
    return 4.34 * (r_m / w_m)**2

def fiber_coupling_eta(fiber: Fiber, w_mode_m: float, dx_m: float, dy_m: float) -> float:
    r = np.sqrt(dx_m**2 + dy_m**2)
    eta = fiber.eta0 * gaussian_overlap_eta_lateral(r, w_mode_m)
    return float(np.clip(eta, 0.0, 1.0))

def fringe_spacing(wavelength_m: float, crossing_angle_rad: float) -> float:
    # delta = lambda / (2 sin(theta/2))
    return wavelength_m / (2.0 * np.sin(crossing_angle_rad/2.0))

def focus_waist_from_beam_diameter(f_m: float, wavelength_m: float, D_e2_m: float) -> float:
    # Given: d_f = 4 f lambda / (pi D_e^-2)  (this is diameter at focus)
    # Convert to waist radius w0 = d/2
    d_f = (4.0 * f_m * wavelength_m) / (PI * D_e2_m)
    return 0.5 * d_f

def fringe_count(d_f_m: float, delta_m: float) -> float:
    return d_f_m / delta_m

def contrast_from_miss_overlap(delta_r_m: float, w_focus_m: float) -> float:
    # Simple model: modulation depth drops with overlap mismatch
    # 1.0 at perfect overlap, decays with lateral miss
    return float(np.exp(-(delta_r_m**2) / (2.0 * w_focus_m**2)))

@dataclass
class PowerStage:
    name: str
    power_w: float
    note: str = ""
