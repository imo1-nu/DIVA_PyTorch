"""
Visualize the stuttering mechanism from DIVA Experiment 2.

Creates a 2-panel figure:
  Top:    F2 trajectory (Hz) vs time — stuttering vs normal DIVA
  Bottom: Auditory error f_size(t) in Hz — stuttering vs normal,
          with repair probability on secondary axis and event markers

Run from repo root:
  UV_PROJECT_ENVIRONMENT=/tmp/diva_venv UV_LINK_MODE=copy \
    uv run --project experiments python experiments/visualize_stutter.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import copy
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from diva_setup import make_diva
from monitoring_subsystem.excessive_error_detector import ExcessiveErrorDetector

FRAME_MS    = 5.0   # ms per DIVA frame
SILENCE_STEPS = 13  # SelfRepairController.SILENCE_STEPS


# ── Simulation with full logging ──────────────────────────────────────────────

def run_logged(diva, detector, seed: int, I_masking: float = 0.0, max_restarts: int = 5):
    """
    Run one production trial, logging formant trajectory and auditory error.

    Returns
    -------
    formant_hz  : (T, 4) array — F0/F1/F2/F3 in Hz (silence = 0)
    fsize_hz    : (T,)   array — total auditory error f_size in Hz
    prep_hz     : (T,)   array — repair probability (0..1)
    events      : list of (frame_idx, label)
                  labels: 'repair_trigger', 'restart'
    n_repairs   : int
    """
    import diva_ssm
    import diva_asm        as aud_sm_mod
    import diva_som_state_map as som_sm_mod
    import diva_aem        as aud_em_mod
    import diva_sem        as som_em_mod

    length_idx = diva.production_info[0].index('length\n')
    total_len  = float(diva.production_info[1][length_idx])
    num_steps  = int(np.ceil(total_len / diva.sample_rate)) + 1

    aud_sm = aud_sm_mod.AuditoryStateMap();      aud_sm.setinputdims(4)
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

    rng       = np.random.default_rng(seed)
    product1  = 1.0
    formant_hz = []
    fsize_hz   = []
    prep_prob  = []
    events     = []
    n_repairs  = 0

    ssm = diva_ssm.SpeechSoundMap(num_steps)

    for attempt in range(max_restarts + 1):
        ssm.LDA.reset(); ssm.LDS.reset(); ssm.dataBuffer = []
        ssm.InputPorts[0] = torch.tensor([product1], dtype=torch.float64)
        ssm.InputPorts[1] = 1
        ssm.output(); ssm.output_seq()

        repair_triggered = False

        for _step in range(num_steps + 10):
            p          = ssm.OutputPorts[0]
            aud_target = ssm.OutputPorts[2]
            som_target = ssm.OutputPorts[3]
            ssm.output_seq()

            aud_em.InputPorts[0] = aud_target
            som_em.InputPorts[0] = som_target
            aud_sm.output();  som_sm.output()
            aud_em.InputPorts[1] = aud_sm.OutputPorts[0]
            som_em.InputPorts[1] = som_sm.OutputPorts[0]
            aud_em.output();  som_em.output()

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

            vt_aud = vocal_tract.OutputPorts[0].numpy().flatten()
            err_np = aud_em.OutputPorts[0].numpy().flatten()
            p_rep  = detector.repair_probability(err_np, I_masking)
            fs_hz  = detector.f_size(err_np)

            formant_hz.append(vt_aud * detector.AUD_SCALE_HZ)
            fsize_hz.append(fs_hz)
            prep_prob.append(p_rep)

            if rng.random() < p_rep:
                events.append((len(formant_hz) - 1, 'repair_trigger'))
                repair_triggered = True
                break

        if not repair_triggered:
            break

        n_repairs += 1

        # Silence gap
        for _ in range(SILENCE_STEPS):
            formant_hz.append(np.zeros(4))
            fsize_hz.append(0.0)
            prep_prob.append(0.0)

        events.append((len(formant_hz), 'restart'))   # first frame of next attempt
        if attempt >= max_restarts:
            break

    return (np.array(formant_hz), np.array(fsize_hz),
            np.array(prep_prob), events, n_repairs)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _shade_silence(ax, events, frame_ms):
    """Grey shading for silence intervals between repair_trigger and restart."""
    for i, (fi, label) in enumerate(events):
        if label == 'repair_trigger':
            t_start = fi * frame_ms
            # find matching restart
            for fj, lj in events[i + 1:]:
                if lj == 'restart':
                    t_end = fj * frame_ms
                    ax.axvspan(t_start, t_end, color='gold', alpha=0.25,
                               zorder=0, label='_nolegend_')
                    break


def plot(fhz_st, fsize_st, prep_st, events_st, n_rep,
         fhz_ns, fsize_ns, prep_ns,
         out_path: str):

    time_st = np.arange(len(fhz_st)) * FRAME_MS
    time_ns = np.arange(len(fhz_ns)) * FRAME_MS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5))
    fig.suptitle(
        'DIVA Stuttering Model: Formant Trajectory & Auditory Error Signal\n'
        '(Civier et al. 2010, Exp. 2 — /bid/ production)',
        fontsize=12
    )

    # ── Panel 1: F2 trajectory ────────────────────────────────────────────────
    ax1.plot(time_ns, fhz_ns[:, 2], color='steelblue', linewidth=1.8,
             label='Normal (ff=0.85, fb=0.30)')
    ax1.plot(time_st, fhz_st[:, 2], color='crimson',   linewidth=1.8,
             label=f'Stutter (ff=0.25, fb=1.50)  repairs={n_rep}')

    _shade_silence(ax1, events_st, FRAME_MS)

    for fi, label in events_st:
        t_ms = fi * FRAME_MS
        if label == 'repair_trigger':
            ax1.axvline(t_ms, color='darkorange', linewidth=1.4,
                        linestyle='--', zorder=3)
            ax1.annotate(
                'Repair trigger\n(error > threshold)', xy=(t_ms, fhz_st[fi, 2]),
                xytext=(t_ms + 15, fhz_st[fi, 2] + 350),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.3),
                fontsize=8.5, color='darkorange', zorder=5
            )
        elif label == 'restart':
            if fi < len(fhz_st):
                ax1.axvline(t_ms, color='seagreen', linewidth=1.4,
                            linestyle='--', zorder=3)
                ax1.annotate(
                    'Restart', xy=(t_ms, fhz_st[fi, 2]),
                    xytext=(t_ms + 10, fhz_st[fi, 2] + 300),
                    arrowprops=dict(arrowstyle='->', color='seagreen', lw=1.3),
                    fontsize=8.5, color='seagreen', zorder=5
                )

    ax1.set_ylabel('F2 (Hz)', fontsize=10)
    ax1.set_xlabel('Time (ms)', fontsize=10)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    legend_patches = [
        Line2D([0], [0], color='steelblue', lw=1.8, label='Normal (ff=0.85, fb=0.30)'),
        Line2D([0], [0], color='crimson',   lw=1.8, label=f'Stutter (ff=0.25, fb=1.50)  repairs={n_rep}'),
        mpatches.Patch(color='gold', alpha=0.5, label='Silent gap (65 ms)'),
    ]
    ax1.legend(handles=legend_patches, fontsize=8.5, loc='lower right')
    ax1.set_title('Second Formant (F2) Trajectory', fontsize=10, pad=4)

    # ── Panel 2: Auditory error f_size ───────────────────────────────────────
    ax2.fill_between(time_ns, fsize_ns, color='steelblue', alpha=0.25)
    ax2.plot(time_ns, fsize_ns, color='steelblue', linewidth=1.5,
             label='Normal')
    ax2.fill_between(time_st, fsize_st, color='crimson', alpha=0.20)
    ax2.plot(time_st, fsize_st, color='crimson', linewidth=1.5,
             label='Stutter')

    _shade_silence(ax2, events_st, FRAME_MS)

    for fi, label in events_st:
        if label == 'repair_trigger':
            t_ms = fi * FRAME_MS
            ax2.axvline(t_ms, color='darkorange', linewidth=1.4,
                        linestyle='--', zorder=3)

    # Secondary y-axis: repair probability
    ax2r = ax2.twinx()
    ax2r.plot(time_st, prep_st * 100, color='darkorange', linewidth=1.2,
              linestyle=':', alpha=0.85, label='Repair prob. (%) [stutter]')
    ax2r.set_ylabel('Repair probability (%)', fontsize=9, color='darkorange')
    ax2r.tick_params(axis='y', labelcolor='darkorange')
    ax2r.set_ylim(bottom=0)

    ax2.set_xlabel('Time (ms)', fontsize=10)
    ax2.set_ylabel('Auditory error f_size (Hz)', fontsize=10)
    ax2.set_title(
        'Auditory error signal  f_size(t) = Σ |dAu_k| × Scale_k\n'
        'Stutter: weak feedforward → large error → repair stochastically triggered',
        fontsize=9, pad=4
    )
    ax2.grid(True, alpha=0.3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building DIVA instances...", flush=True)
    diva_st = make_diva('bid', 0.25, 1.50, learning_phase_runs=4, seed=42)
    diva_ns = make_diva('bid', 0.85, 0.30, learning_phase_runs=4, seed=42)
    print("Done.", flush=True)

    detector = ExcessiveErrorDetector(epsilon=2e-5)

    # Find a stuttering trial with at least 1 repair
    print("Finding a stuttering trial with repair...", flush=True)
    found_seed = None
    for seed in range(200):
        fhz_st, fsize_st, prep_st, events_st, n_rep = run_logged(diva_st, detector, seed)
        if n_rep >= 1:
            found_seed = seed
            print(f"  seed={seed}  n_repairs={n_rep}  frames={len(fhz_st)}", flush=True)
            break

    if found_seed is None:
        print("  No repair in 200 seeds; using seed=0 (may have 0 repairs).", flush=True)
        fhz_st, fsize_st, prep_st, events_st, n_rep = run_logged(diva_st, detector, 0)

    # Normal trial
    print("Running normal trial (seed=0)...", flush=True)
    fhz_ns, fsize_ns, prep_ns, events_ns, _ = run_logged(diva_ns, detector, 0)

    # ── Print statistics ──────────────────────────────────────────────────────
    print("\n--- 数値要約 ---", flush=True)
    print(f"正常発話: {len(fhz_ns)} フレーム = {len(fhz_ns)*FRAME_MS:.0f} ms")
    print(f"  f_size(max)={fsize_ns.max():.1f} Hz  p_rep(max)={prep_ns.max()*100:.3f}%")
    print(f"吃音発話: {len(fhz_st)} フレーム = {len(fhz_st)*FRAME_MS:.0f} ms  修復={n_rep}回")
    print(f"  f_size(max)={fsize_st.max():.1f} Hz  p_rep(max)={prep_st.max()*100:.3f}%")
    for fi, label in events_st:
        t_ms = fi * FRAME_MS
        print(f"  イベント [{label}] フレーム={fi}  t={t_ms:.0f} ms"
              + (f"  f_size={fsize_st[fi]:.1f} Hz" if fi < len(fsize_st) else ""))

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'stutter_visualization.png')
    print(f"\nPlotting...", flush=True)
    plot(fhz_st, fsize_st, prep_st, events_st, n_rep,
         fhz_ns, fsize_ns, prep_ns,
         out_path)


if __name__ == '__main__':
    main()
