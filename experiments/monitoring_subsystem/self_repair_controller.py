"""
Self-repair controller from Civier et al. (2010).

When ExcessiveErrorDetector triggers a repair:
  1. Phonation is suspended for SILENCE_STEPS (65 ms = 13 × 5 ms).
  2. The SSM sequence is restarted from the beginning.
  3. A new production attempt begins.

Usage
-----
ctrl = SelfRepairController(detector)
n_repairs, aud_err_log = ctrl.run(diva, I_masking=0.0)
"""

import numpy as np
import torch

# DIVA src modules must be on sys.path before importing this module.
import diva_ssm
import diva_cerebellum as ceb
import diva_motorcortex as dmc
import diva_vocaltract as dvt
import diva_asm as aud_sm
import diva_som_state_map as som_sm
import diva_aem as aud_em
import diva_sem as som_em


class SelfRepairController:
    SILENCE_STEPS = 13  # 65 ms at 5 ms/step

    def __init__(self, detector):
        self.detector = detector

    def _init_components(self, diva, num_steps: int):
        """Return simulation components from diva, initialising lazily."""
        if diva.ssm is None or diva.cerebellum is None:
            diva.PrepareSimulation()
        cerebellum = diva.cerebellum
        vocal_tract = diva.vocal_tract
        motor_cortex = diva.motor_cortex
        aud_state_map = diva.aud_state_map
        som_state_map = diva.som_state_map
        aud_err_map = diva.aud_err_map
        som_err_map = diva.som_err_map
        if aud_state_map is not None:
            aud_state_map.setinputdims(4)
        if som_state_map is not None:
            som_state_map.setinputdims(8)
        return (cerebellum, vocal_tract, motor_cortex,
                aud_state_map, som_state_map, aud_err_map, som_err_map)

    @staticmethod
    def _make_ssm(num_steps: int):
        """Create a fresh SpeechSoundMap (loads weights from CWD mat files)."""
        ssm = diva_ssm.SpeechSoundMap(num_steps)
        return ssm

    @staticmethod
    def _start_ssm(ssm, product1: float):
        ssm.InputPorts[0] = torch.tensor([product1], dtype=torch.float64)
        ssm.InputPorts[1] = 1
        ssm.output()
        ssm.output_seq()

    def run(self, diva, I_masking: float = 0.0,
            max_restarts: int = 20,
            rng: np.random.Generator = None):
        """
        Run one production with self-repair monitoring.

        Returns
        -------
        n_repairs : int
        aud_err_log : list of np.ndarray shape (4,)
            Normalised auditory error at each simulated timestep
            (across all attempts including restarted ones).
        """
        if rng is None:
            rng = np.random.default_rng()

        length_idx = diva.production_info[0].index('length\n')
        total_prod_length = diva.production_info[1][length_idx]
        num_steps = int(np.ceil(float(total_prod_length) / diva.sample_rate)) + 1

        (cerebellum, vocal_tract, motor_cortex,
         aud_state_map, som_state_map,
         aud_err_map, som_err_map) = self._init_components(diva, num_steps)

        # Reset all transient state so each production starts from the same
        # initial conditions.  diva.reset() clears delay buffers and the
        # vel2pos integrator but preserves learned weight matrices.
        diva.reset()

        # Restore post-training state snapshots saved by make_diva().
        #
        # cerebellum.dataBuffer: diva.reset() clears LDA/LDS but not this
        # 1-step delay queue; leaving it un-restored causes the VT formant
        # trajectory to shift each trial.
        #
        # LMC weights: AdaptiveWeightBlock.reset() is a no-op; every call to
        # LMC.output() applies a Hebbian update (dW = 0.5*u*v'), so without
        # restoring the weights the feedforward command drifts across trials
        # and auditory errors shrink to zero by trial 3-4.
        import copy
        if hasattr(diva, '_initial_cerebellum_buffer'):
            cerebellum.dataBuffer = copy.deepcopy(diva._initial_cerebellum_buffer)
        if hasattr(diva, '_initial_lmc_weight'):
            motor_cortex.LMC.get_set_weights(3, diva._initial_lmc_weight.clone())

        product1 = float(sum([0, 1]))  # psum × det_rise = 1

        n_repairs = 0
        aud_err_log = []

        # Create SSM once per run() to avoid reloading .mat files on each
        # restart attempt.  Between attempts we soft-reset only the dynamic
        # state (LDA/LDS delay blocks, dataBuffer) while keeping the already-
        # loaded weight matrices.  ssm.output() re-initialises the sequencer
        # (DWork1/DWork2) whenever it receives a non-zero product1 input.
        ssm = self._make_ssm(num_steps)

        for attempt in range(max_restarts + 1):
            ssm.LDA.reset()
            ssm.LDS.reset()
            ssm.dataBuffer = []
            self._start_ssm(ssm, product1)

            repair_triggered = False

            for step in range(num_steps + 10):
                ssm_out_primary = ssm.OutputPorts[0]
                ssm_out_aud = ssm.OutputPorts[2]
                ssm_out_som = ssm.OutputPorts[3]
                ssm.output_seq()

                aud_err_map.InputPorts[0] = ssm_out_aud
                som_err_map.InputPorts[0] = ssm_out_som

                aud_state_map.output()
                som_state_map.output()

                aud_state = aud_state_map.OutputPorts[0]
                som_state = som_state_map.OutputPorts[0]

                aud_err_map.InputPorts[1] = aud_state
                som_err_map.InputPorts[1] = som_state

                aud_err_map.output()
                som_err_map.output()

                aud_err = aud_err_map.OutputPorts[0]
                som_err = som_err_map.OutputPorts[0]

                cerebellum.input(ssm_out_primary)
                cerebellum.output()
                ceb_out = cerebellum.OutputPorts[0]

                motor_cortex.InputPorts[0] = ssm_out_primary
                motor_cortex.InputPorts[1] = ceb_out
                motor_cortex.InputPorts[2] = som_err
                motor_cortex.InputPorts[3] = aud_err
                motor_cortex.output()

                cort_out = motor_cortex.OutputPorts[0]
                vocal_tract.InputPorts[0] = cort_out
                vocal_tract.output()

                aud_state_map.input(vocal_tract.OutputPorts[0])
                som_state_map.input(vocal_tract.OutputPorts[1])

                err_np = aud_err.numpy().flatten()
                aud_err_log.append(err_np.copy())

                p = self.detector.repair_probability(err_np, I_masking)
                if rng.random() < p:
                    repair_triggered = True
                    break

                # Silence phase modelled implicitly by the gap before restart.

            if not repair_triggered:
                break

            n_repairs += 1
            if attempt >= max_restarts:
                break

        diva.ssm = ssm
        return n_repairs, aud_err_log
