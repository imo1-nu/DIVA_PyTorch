"""
Synthesize audio of a stuttering repetition from DIVA Experiment 2.

Steps:
  1. Run SelfRepairController on stuttering DIVA until a trial with repair is found.
  2. Record VT formant output + voicing at each step (including silence + restart).
  3. Synthesize using a Klatt-style cascade formant synthesizer.
  4. Save to stuttering_repetition.wav (and normal_bid.wav for comparison).

Run from repo root:
  UV_PROJECT_ENVIRONMENT=/tmp/diva_venv UV_LINK_MODE=copy \\
    uv run --project experiments python experiments/synthesize_stuttering.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import copy
import numpy as np
import torch
from scipy import signal as sig
import soundfile as sf

from diva_setup import make_diva
from monitoring_subsystem.excessive_error_detector import ExcessiveErrorDetector
from monitoring_subsystem.self_repair_controller import SelfRepairController

# ── Constants ────────────────────────────────────────────────────────────────

AUDIO_SR    = 22050        # audio sample rate (Hz)
DIVA_FRAME  = 0.005        # DIVA frame duration (s)
AUDIO_PER_FRAME = int(AUDIO_SR * DIVA_FRAME)  # samples per DIVA frame

# DIVA auditory output scale: 1 normalized unit → X Hz
AUD_SCALE_HZ = np.array([100.0, 500.0, 1500.0, 3000.0])  # F0, F1, F2, F3

SILENCE_STEPS = SelfRepairController.SILENCE_STEPS   # 13 steps = 65 ms

# ── Formant synthesizer ───────────────────────────────────────────────────────

def _resonator_coeffs(freq_hz: float, bw_hz: float, sr: int):
    """Second-order resonator (Klatt 1980 cascade filter)."""
    T = 1.0 / sr
    C = -np.exp(-2 * np.pi * bw_hz * T)
    B =  2 * np.exp(-np.pi  * bw_hz * T) * np.cos(2 * np.pi * freq_hz * T)
    A = 1.0 - B - C
    return np.array([A]), np.array([1.0, -B, -C])


def _glottal_source(f0_hz: float, n_samples: int, sr: int) -> np.ndarray:
    """Periodic impulse train at F0 (voiced) or white noise (unvoiced)."""
    if f0_hz < 20:
        return np.random.default_rng(42).normal(0, 0.3, n_samples)
    period = int(sr / f0_hz)
    x = np.zeros(n_samples)
    x[::period] = 1.0
    return x


def synthesize_frames(formant_traj: np.ndarray,
                      voiced_traj: np.ndarray,
                      sr: int = AUDIO_SR,
                      samples_per_frame: int = AUDIO_PER_FRAME) -> np.ndarray:
    """
    Convert formant trajectory to audio.

    Parameters
    ----------
    formant_traj : (T, 4) — F0, F1, F2, F3 in Hz
    voiced_traj  : (T,)   — voicing flag (>0.5 = voiced)
    """
    # Per-formant bandwidths (Hz) — simplified
    BANDWIDTHS = [30.0, 80.0, 100.0, 120.0]   # BW for F0, F1, F2, F3

    n_frames = len(formant_traj)
    audio = np.zeros(n_frames * samples_per_frame)

    # IIR filter states (for continuity across frames)
    zi = [np.zeros(2) for _ in range(4)]

    for t in range(n_frames):
        f = formant_traj[t]          # [F0, F1, F2, F3] Hz
        voiced = voiced_traj[t] > 0.5

        # Glottal source for this frame
        src = _glottal_source(f[0] if voiced else 0.0,
                               samples_per_frame, sr)

        # Cascade resonators (F1 → F2 → F3; skip F0 resonator)
        out = src.copy()
        for k in range(1, 4):
            fq = max(f[k], 80.0)     # clamp to avoid instability
            b, a = _resonator_coeffs(fq, BANDWIDTHS[k], sr)
            out, zi[k] = sig.lfilter(b, a, out, zi=zi[k])

        # Apply gentle amplitude envelope within frame
        env = np.hanning(samples_per_frame)
        out = out * env * 0.4

        start = t * samples_per_frame
        audio[start:start + samples_per_frame] = out

    # Normalise
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    return (audio * 0.9 * 32767).astype(np.int16)


# ── DIVA simulation with formant logging ──────────────────────────────────────

def run_with_formant_log(diva, detector, rng,
                         I_masking: float = 0.0,
                         force_repair: bool = False,
                         max_restarts: int = 5):
    """
    Like SelfRepairController.run() but also records VT formant output.

    Returns
    -------
    n_repairs : int
    formant_log : list of np.ndarray (4,) in Hz  — full timeline incl. silence/restarts
    voiced_log  : list of float — voicing flag per frame
    """
    import diva_ssm
    import diva_asm as aud_sm_mod
    import diva_som_state_map as som_sm_mod
    import diva_aem as aud_em_mod
    import diva_sem as som_em_mod

    length_idx = diva.production_info[0].index('length\n')
    total_len  = float(diva.production_info[1][length_idx])
    num_steps  = int(np.ceil(total_len / diva.sample_rate)) + 1

    # Fresh state maps (independent of diva's stored ones)
    aud_sm = aud_sm_mod.AuditoryStateMap();  aud_sm.setinputdims(4)
    som_sm = som_sm_mod.SomatosensoryStateMap(); som_sm.setinputdims(8)
    aud_em = aud_em_mod.AuditoryErrorMap()
    som_em = som_em_mod.SomatosensoryErrorMap()

    cerebellum  = diva.cerebellum
    vocal_tract = diva.vocal_tract
    mc          = diva.motor_cortex

    diva.reset()
    if hasattr(diva, '_initial_cerebellum_buffer'):
        cerebellum.dataBuffer = copy.deepcopy(diva._initial_cerebellum_buffer)
    if hasattr(diva, '_initial_lmc_weight'):
        mc.LMC.get_set_weights(3, diva._initial_lmc_weight.clone())

    product1 = 1.0
    formant_log = []
    voiced_log  = []
    n_repairs   = 0

    # SSM created once; soft-reset between attempts
    ssm = diva_ssm.SpeechSoundMap(num_steps)

    for attempt in range(max_restarts + 1):
        ssm.LDA.reset(); ssm.LDS.reset(); ssm.dataBuffer = []
        ssm.InputPorts[0] = torch.tensor([product1], dtype=torch.float64)
        ssm.InputPorts[1] = 1
        ssm.output(); ssm.output_seq()

        repair_triggered = False

        for step in range(num_steps + 10):
            p          = ssm.OutputPorts[0]
            aud_target = ssm.OutputPorts[2]
            som_target = ssm.OutputPorts[3]
            ssm.output_seq()

            aud_em.InputPorts[0] = aud_target
            som_em.InputPorts[0] = som_target

            aud_sm.output(); som_sm.output()
            aud_em.InputPorts[1] = aud_sm.OutputPorts[0]
            som_em.InputPorts[1] = som_sm.OutputPorts[0]
            aud_em.output(); som_em.output()

            cerebellum.input(p); cerebellum.output()
            mc.InputPorts[0] = p
            mc.InputPorts[1] = cerebellum.OutputPorts[0]
            mc.InputPorts[2] = som_em.OutputPorts[0]
            mc.InputPorts[3] = aud_em.OutputPorts[0]
            mc.output()

            vocal_tract.InputPorts[0] = mc.OutputPorts[0]
            vocal_tract.output()

            aud_sm.input(vocal_tract.OutputPorts[0])
            som_sm.input(vocal_tract.OutputPorts[1])

            # Record formants (OutputPorts[0] = auditory, normalised)
            vt_aud = vocal_tract.OutputPorts[0].numpy().flatten()
            formant_log.append(vt_aud * AUD_SCALE_HZ)        # convert to Hz
            # Voicing: use pressure/voicing flag from somatosensory output
            voiced_log.append(1.0 if vt_aud[0] > 0.05 else 0.0)

            err_np = aud_em.OutputPorts[0].numpy().flatten()
            p_rep  = detector.repair_probability(err_np, I_masking)

            triggered = rng.random() < p_rep if not force_repair else (
                step > 5 and attempt == 0   # force repair on first attempt after step 5
            )
            if triggered:
                repair_triggered = True
                break

        if not repair_triggered:
            break

        n_repairs += 1
        # Insert silence
        silence = np.zeros(4)
        for _ in range(SILENCE_STEPS):
            formant_log.append(silence)
            voiced_log.append(0.0)

        if attempt >= max_restarts:
            break

    return n_repairs, np.array(formant_log), np.array(voiced_log)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building DIVA instances...", flush=True)
    diva_st = make_diva('bid', 0.25, 1.50, learning_phase_runs=4, seed=42)
    diva_ns = make_diva('bid', 0.85, 0.30, learning_phase_runs=4, seed=42)
    print("Done.", flush=True)

    detector = ExcessiveErrorDetector(epsilon=2e-5)

    # ── Stuttering: find a trial with at least 1 repair ──────────────────────
    print("Looking for a stuttering trial with repair...", flush=True)
    rng = np.random.default_rng(0)
    for seed in range(200):
        rng_t = np.random.default_rng(seed)
        n, flog, vlog = run_with_formant_log(diva_st, detector, rng_t)
        if n >= 1:
            print(f"  Found: seed={seed}, n_repairs={n}, frames={len(flog)}", flush=True)
            break
    else:
        print("  No repair found in 200 seeds; using forced repair.", flush=True)
        rng_t = np.random.default_rng(0)
        n, flog, vlog = run_with_formant_log(diva_st, detector, rng_t,
                                              force_repair=True)

    # Synthesize stuttering audio
    print("Synthesising stuttering audio...", flush=True)
    # F0 fixed at ~110 Hz (male voice) throughout — DIVA F0 output is always near 0
    flog_fixed = flog.copy()
    flog_fixed[:, 0] = np.where(vlog > 0.5, 110.0, 0.0)
    wav_stutter = synthesize_frames(flog_fixed, vlog)
    out_stutter = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'stuttering_repetition.wav')
    sf.write(out_stutter, wav_stutter, AUDIO_SR, subtype='PCM_16')
    print(f"  Saved: {out_stutter}  ({len(flog)} frames = {len(flog)*5} ms)", flush=True)

    # ── Normal: one clean trial ───────────────────────────────────────────────
    print("Synthesising normal (fluent) audio...", flush=True)
    rng_ns = np.random.default_rng(0)
    _, flog_ns, vlog_ns = run_with_formant_log(diva_ns, detector, rng_ns)
    flog_ns_fixed = flog_ns.copy()
    flog_ns_fixed[:, 0] = np.where(vlog_ns > 0.5, 110.0, 0.0)
    wav_normal = synthesize_frames(flog_ns_fixed, vlog_ns)
    out_normal = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'normal_bid.wav')
    sf.write(out_normal, wav_normal, AUDIO_SR, subtype='PCM_16')
    print(f"  Saved: {out_normal}  ({len(flog_ns)} frames = {len(flog_ns)*5} ms)", flush=True)

    # ── Print formant summary ─────────────────────────────────────────────────
    print("\n--- Formant summary (Hz) ---")
    print(f"  Stuttering  F2 mean={flog[:,2].mean():.0f} Hz  max={flog[:,2].max():.0f} Hz")
    print(f"  Normal      F2 mean={flog_ns[:,2].mean():.0f} Hz  max={flog_ns[:,2].max():.0f} Hz")
    print(f"  n_repairs (stuttering trial) = {n}", flush=True)


if __name__ == '__main__':
    main()
