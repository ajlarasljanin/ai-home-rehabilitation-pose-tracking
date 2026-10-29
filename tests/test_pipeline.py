"""
test_pipeline.py

Jedinicni testovi jezgra (ugao, filter, FSM) na sintetickim signalima.

    pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest

from pose_pipeline.calibration import AnthropometricCalibrator
from pose_pipeline.filtering import (
    ButterworthCausal,
    FilterMode,
    KinematicFilter,
    moving_average,
    zero_phase_filter,
)
from pose_pipeline.fsm import FSMConfig, SquatFSM, SquatState
from pose_pipeline.interpolation import linear_interpolate_invalid
from pose_pipeline.kinematics import causal_velocity_step, central_difference_velocity, linear_velocity
from pose_pipeline.landmarks import Landmark, joint_angle, segment_length, shoulder_angle, visibility_gate
from pose_pipeline.segmentation import detect_body_momentum


# ---------------------------------------------------------------------------
# landmarks.py -- jednacine (1)-(6), sekcija 3.1
# ---------------------------------------------------------------------------

def test_joint_angle_right_angle():
    """Tri tacke koje geometrijski formiraju tacno 90 stepeni."""
    vertex = Landmark(0, 0, 0, 1.0)
    a = Landmark(1, 0, 0, 1.0)   # vektor (1,0,0)
    b = Landmark(0, 1, 0, 1.0)   # vektor (0,1,0), ortogonalan na prvi
    assert joint_angle(a, vertex, b) == pytest.approx(90.0, abs=1e-6)


def test_joint_angle_straight_leg():
    """Kolena u punoj ekstenziji: kuk-koleno-skocni zglob u pravoj liniji -> ~180 stepeni."""
    hip = Landmark(0, 0, 0, 1.0)
    knee = Landmark(0, 1, 0, 1.0)
    ankle = Landmark(0, 2, 0, 1.0)
    assert joint_angle(hip, knee, ankle) == pytest.approx(180.0, abs=1e-6)


def test_joint_angle_numerical_clip_stability():
    """arccos mora ostati definisan i kad je cos(theta) numericki malo van [-1, 1]."""
    vertex = Landmark(0, 0, 0, 1.0)
    # Skoro paralelni vektori razlicite duzine -> cos(theta) blizu 1
    a = Landmark(1.0, 1e-12, 0, 1.0)
    b = Landmark(2.0, -1e-12, 0, 1.0)
    theta = joint_angle(a, vertex, b)
    assert 0.0 <= theta <= 1.0  # skoro kolinearni -> ugao blizu 0


def test_segment_length_euclidean():
    p1 = Landmark(0, 0, 0, 1.0)
    p2 = Landmark(3, 4, 0, 1.0)
    assert segment_length(p1, p2) == pytest.approx(5.0)


def test_visibility_gate_blocks_low_confidence():
    lm = {1: Landmark(0, 0, 0, visibility=0.9), 2: Landmark(0, 0, 0, visibility=0.2)}
    assert visibility_gate(lm, [1, 2]) is False
    assert visibility_gate(lm, [1]) is True


# ---------------------------------------------------------------------------
# filtering.py -- Butterworth (sekcija 3.3) i MA (sekcija 3.3.2)
# ---------------------------------------------------------------------------

def test_butterworth_causal_attenuates_high_frequency_noise():
    """
    Sintet. signal: spor 'pokret' (0.5 Hz sinusoida) + brz sum (12 Hz), fs=30Hz.
    Butterworth (fc=6Hz) treba znacajno da potisne 12 Hz komponentu, sekcija 3.3.1.
    """
    fs = 30.0
    t = np.arange(0, 4, 1 / fs)
    slow_signal = 20 * np.sin(2 * np.pi * 0.5 * t) + 130  # simulira ugao kolena
    noise = 5 * np.sin(2 * np.pi * 12 * t)
    noisy = slow_signal + noise

    filt = ButterworthCausal(cutoff_hz=6.0, sample_rate_hz=fs, order=2)
    filtered = filt.filter_sequence_causal(noisy)

    # Nakon prelaznog perioda filtera, izlaz treba da bude blizi cistom signalu
    # nego sto je bio zasumljeni ulaz (mereno u stabilnom delu signala).
    steady = slice(60, None)
    err_before = np.mean(np.abs(noisy[steady] - slow_signal[steady]))
    err_after = np.mean(np.abs(filtered[steady] - slow_signal[steady]))
    assert err_after < 0.5 * err_before


def test_zero_phase_filter_preserves_extremum_timing():
    """
    Sekcija 3.3.1: zero-phase filtriranje ne sme pomeriti poziciju ekstremuma
    u vremenu (za razliku od kauzalnog filtera koji uvodi fazni pomak).
    """
    fs = 30.0
    t = np.arange(0, 2, 1 / fs)
    signal = -20 * np.cos(2 * np.pi * 1.0 * t) + 130  # minimum tacno na t=0.5s (indeks 15)
    noisy = signal + np.random.default_rng(42).normal(0, 0.5, size=signal.shape)

    filtered = zero_phase_filter(noisy, cutoff_hz=6.0, sample_rate_hz=fs)
    true_min_idx = int(np.argmin(signal[:30]))
    filtered_min_idx = int(np.argmin(filtered[:30]))
    assert abs(filtered_min_idx - true_min_idx) <= 1  # tolerancija +/- 1 frejm


def test_moving_average_matches_manual_formula():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = moving_average(x, window=3)
    # sredisnji element prozora [3,4,5] treba da bude (3+4+5)/3 = 4.0
    idx_of_value_4 = 3
    assert result[idx_of_value_4] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# kinematics.py -- centralna vs. kauzalna razlika (sekcije 3.2.1 i 5.2.1)
# ---------------------------------------------------------------------------

def test_central_difference_more_accurate_than_causal_on_smooth_signal():
    """
    Za glatku funkciju theta(t) = sin(t), tacan izvod je cos(t). Centralna
    razlika (O(dt^2)) treba da bude tacnija od kauzalne, unazadne (O(dt)) --
    ovo je numericko opravdanje teksta u sekciji 5.2.1.
    """
    dt = 0.05
    t = np.arange(0, 10, dt)
    theta = np.sin(t)
    true_omega = np.cos(t)

    omega_central = central_difference_velocity(theta, dt)
    omega_causal = np.array([
        causal_velocity_step(theta[i], theta[i - 1], dt) for i in range(1, len(theta))
    ])

    err_central = np.mean(np.abs(omega_central[5:-5] - true_omega[5:-5]))
    err_causal = np.mean(np.abs(omega_causal[4:-5] - true_omega[5:-5]))
    assert err_central < err_causal


def test_linear_velocity_of_static_point_is_zero():
    positions = np.tile(np.array([0.5, 0.5, 0.0]), (50, 1))
    v = linear_velocity(positions, dt=1 / 30)
    assert np.allclose(v, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# fsm.py -- Algoritam 1, histereza, Majority-Vote, refraktorni period
# ---------------------------------------------------------------------------

def _synthetic_squat_sequence(n_reps: int, fs: float = 30.0,
                               rep_duration_s: float = 1.5,
                               depth_deg: float = 70.0,
                               descend_frac: float = 0.35,
                               hold_frac: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """Sinteticki cucanj: spust, kratko zadrzavanje na dnu, pa podizanje."""
    dt = 1 / fs
    theta_ref = 175.0
    theta_min = theta_ref - depth_deg  # dno cucnja; depth_deg=100 -> theta_min=75 stepeni

    total_samples = int(rep_duration_s / dt)
    descend_n = int(total_samples * descend_frac)
    hold_n = int(total_samples * hold_frac)
    ascend_n = total_samples - descend_n - hold_n

    descend = theta_ref - (theta_ref - theta_min) * (
        0.5 - 0.5 * np.cos(np.linspace(0, np.pi, descend_n))
    )
    hold = np.full(hold_n, theta_min)
    ascend = theta_min + (theta_ref - theta_min) * (
        0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ascend_n))
    )
    one_rep = np.concatenate([descend, hold, ascend])

    # kratka pauza (mirovanje u ekstenziji) izmedju ponavljanja, da FSM ima
    # vremena da se vrati u S_start pre sledeceg cucnja
    pause = np.full(int(0.3 * fs), theta_ref)
    theta_full = []
    for _ in range(n_reps):
        theta_full.append(one_rep)
        theta_full.append(pause)
    theta_full = np.concatenate(theta_full)
    t = np.arange(len(theta_full)) * dt
    return theta_full, t


def test_fsm_counts_correct_number_of_clean_squats():
    theta, t = _synthetic_squat_sequence(n_reps=3, depth_deg=100.0)
    fsm = SquatFSM(FSMConfig())  # eps_omega=15 deg/s (sekcija 5.4 / specifikacija)

    for i, (theta_t, ts) in enumerate(zip(theta, t)):
        fsm.step(theta_t, timestamp=float(ts), g_valid=True, frame_index=i)

    assert fsm.repetition_counter == 3


def test_fsm_rejects_shallow_squat():
    """Plitak cucanj (ne prelazi theta_down - eps_hyst) ne sme se izbrojati (sekcija 5.5.1)."""
    fs = 30.0
    dt = 1 / fs
    t = np.arange(0, 3, dt)
    theta_ref = 175.0
    # Plitak pokret: ide samo do 130 stepeni, sto je iznad theta_down=100
    # (a kamoli theta_down - eps_hyst = 90), pa HOLD nikad ne sme da se potvrdi.
    theta = theta_ref - 45 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 3.0))

    fsm = SquatFSM(FSMConfig(eps_omega=15.0))
    for i, (theta_t, ts) in enumerate(zip(theta, t)):
        fsm.step(theta_t, timestamp=float(ts), g_valid=True, frame_index=i)

    assert fsm.repetition_counter == 0


def test_fsm_hysteresis_prevents_bouncing_near_threshold():
    """
    Signal koji oscilira TACNO oko theta_down (100 st.) bez ikad dostizanja
    theta_down - eps_hyst ne sme uzrokovati lazno brojanje (motivacija za
    histerezu, sekcija 5.3.1 -- "sistem ce da poludi i broji 50 puta").
    """
    fs = 30.0
    dt = 1 / fs
    n = int(5 * fs)
    rng = np.random.default_rng(0)
    # oscilacija između 99 i 101 stepeni oko praga od 100 -- NIKAD ne prelazi
    # donji histerezicni prag theta_down - eps_hyst = 90
    theta = 100.0 + rng.normal(0, 0.6, size=n)

    fsm = SquatFSM(FSMConfig(eps_omega=15.0))
    for i in range(n):
        fsm.step(float(theta[i]), timestamp=i * dt, g_valid=True, frame_index=i)

    assert fsm.repetition_counter == 0


def test_fsm_refractory_period_blocks_double_count():
    """
    Dva potpuna ciklusa odvojena vremenskim razmakom KRACIM od T_refractory
    ne smeju oba biti izbrojana (sekcija 5.5.2, T_refractory=0.5s).
    """
    fs = 30.0
    dt = 1 / fs
    theta_ref = 175.0
    theta_min = 75.0
    samples_per_rep = int(0.2 * fs)  # veoma brz cucanj, ~200ms po ciklusu

    one_rep = theta_ref - (theta_ref - theta_min) * (
        0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, samples_per_rep))
    )
    theta = np.concatenate([one_rep, one_rep])  # dva brza ciklusa odmah jedan za drugim
    t = np.arange(len(theta)) * dt

    fsm = SquatFSM(FSMConfig(eps_omega=50.0, t_min_hold=0.02))
    for i, (theta_t, ts) in enumerate(zip(theta, t)):
        fsm.step(float(theta_t), timestamp=float(ts), g_valid=True, frame_index=i)

    # Refraktorni period treba da spreci da se DRUGI, prebrzi ciklus izbroji.
    assert fsm.repetition_counter <= 1


def test_fsm_ignores_frames_with_low_visibility():
    """Kada je G_valid=False, FSM ne sme da menja stanje (sekcija 5, 7.3)."""
    fsm = SquatFSM(FSMConfig(eps_omega=15.0))
    fsm.step(175.0, timestamp=0.0, g_valid=True, frame_index=0)
    state_before = fsm.state
    # Vrednost koja bi inace pokrenula tranziciju u DOWN, ali sa g_valid=False:
    fsm.step(50.0, timestamp=0.1, g_valid=False, frame_index=1)
    assert fsm.state == state_before


# ---------------------------------------------------------------------------
# calibration.py
# ---------------------------------------------------------------------------

def test_calibrator_requires_minimum_samples_before_result():
    cal = AnthropometricCalibrator(min_samples=10)
    lm = {
        11: Landmark(0.5, 0.0, 0.0, 0.9),   # rame
        23: Landmark(0.5, 0.5, 0.0, 0.9),   # kuk
        25: Landmark(0.5, 0.9, 0.0, 0.9),   # koleno
        27: Landmark(0.5, 1.3, 0.0, 0.9),   # skocni zglob
    }
    for _ in range(5):
        cal.add_frame(lm)
    assert not cal.is_calibrated
    with pytest.raises(RuntimeError):
        cal.result()

    for _ in range(10):
        cal.add_frame(lm)
    assert cal.is_calibrated
    res = cal.result()
    assert res["thigh_length"] == pytest.approx(0.4, abs=1e-6)
    assert res["shank_length"] == pytest.approx(0.4, abs=1e-6)


def test_calibrator_rejects_low_visibility_frames():
    cal = AnthropometricCalibrator(min_samples=5)
    low_confidence_lm = {
        11: Landmark(0, 0, 0, 0.9),
        23: Landmark(0, 0.5, 0, 0.9),
        25: Landmark(0, 0.9, 0, 0.3),  # koleno ispod praga -> ceo frejm se odbacuje
        27: Landmark(0, 1.3, 0, 0.9),
    }
    for _ in range(20):
        cal.add_frame(low_confidence_lm)
    assert cal.n_valid_samples == 0


# ---------------------------------------------------------------------------
# interpolacija Visibility Score + KinematicFilter rezimi
# ---------------------------------------------------------------------------

def test_linear_interpolate_fills_invalid_gap():
    theta = np.array([0.0, np.nan, np.nan, 3.0])
    valid = np.array([True, False, False, True])
    filled = linear_interpolate_invalid(theta, valid)
    assert filled[1] == pytest.approx(1.0)
    assert filled[2] == pytest.approx(2.0)


def test_kinematic_filter_realtime_rejects_offline_api():
    filt = KinematicFilter(mode=FilterMode.REALTIME)
    with pytest.raises(ValueError):
        filt.process_sequence(np.array([1.0, 2.0, 3.0]))


def test_kinematic_filter_offline_uses_central_difference():
    fs = 30.0
    dt = 1 / fs
    t = np.arange(0, 2, dt)
    theta = 90.0 + 20.0 * np.sin(2 * np.pi * 0.5 * t)
    filt = KinematicFilter(mode=FilterMode.OFFLINE, sample_rate_hz=fs)
    kin = filt.process_sequence(theta)
    omega_ref = central_difference_velocity(kin.theta_filtered, dt)
    assert np.allclose(kin.omega, omega_ref)


def test_shoulder_angle_and_body_momentum():
    """theta_shoulder = ∠(L_11, L_13, L_23); oscilacija > 10 deg = body momentum."""
    from pose_pipeline.landmarks import LEFT_ELBOW, LEFT_HIP, LEFT_SHOULDER

    def lm(sx, sy, ex, ey):
        return {
            LEFT_SHOULDER: Landmark(sx, sy, 0, 1.0),
            LEFT_ELBOW: Landmark(ex, ey, 0, 1.0),
            LEFT_HIP: Landmark(0.5, 0.8, 0, 1.0),
        }

    a1 = shoulder_angle(lm(0.5, 0.2, 0.7, 0.2))
    a2 = shoulder_angle(lm(0.5, 0.2, 0.7, 0.45))
    assert detect_body_momentum(np.array([a1, a1, a2]), threshold_deg=10.0) is True
    assert detect_body_momentum(np.array([a1, a1 + 2.0]), threshold_deg=10.0) is False


def test_majority_vote_requires_four_of_five_frames():
    """Jedan uzorak ispod theta_down ne sme da okine START -> DOWN (N=5, delta=4/5)."""
    fs = 30.0
    dt = 1 / fs
    fsm = SquatFSM(FSMConfig())
    fsm.step(175.0, timestamp=0.0, g_valid=True, frame_index=0)
    fsm.step(50.0, timestamp=dt, g_valid=True, frame_index=1)
    assert fsm.state == SquatState.START
    for i in range(4):
        fsm.step(50.0, timestamp=(i + 2) * dt, g_valid=True, frame_index=i + 2)
    assert fsm.state == SquatState.DOWN
