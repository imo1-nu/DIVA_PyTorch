"""
Shared DIVA setup utilities for experiment scripts.

Usage
-----
import sys, os
sys.path.insert(0, '<path>/src')
sys.path.insert(0, '<path>/experiments')
os.chdir('<path>/src')

from diva_setup import make_diva
diva = make_diva('bid', feedforward_gain=0.25, feedback_gain=1.50,
                 learning_phase_runs=4)
"""

import os
import numpy as np
import torch


def setup_target(diva, target_label: str, length_override: float = None) -> None:
    """
    Load a named target into an existing DIVA instance.

    Parameters
    ----------
    diva : DIVA
    target_label : str
        Label as it appears in diva.csv (e.g. 'bid', 'bad', 'good_dog').
    length_override : float, optional
        Override target length in ms (used for slow-speech experiments).
    """
    idx = diva.production_labels.index(target_label)
    id_ = diva.production_ids[idx]
    diva.production_idx = idx
    diva.targetIndex = idx
    diva.production_info = diva.LoadTarget('txt', id_)

    if length_override is not None:
        li = diva.production_info[0].index('length\n')
        diva.production_info[1][li] = str(length_override) + '\n'

    diva.production_art = None
    for attr in ('motor_cortex', 'ssm', 'cerebellum', 'vocal_tract',
                 'aud_state_map', 'som_state_map', 'aud_err_map', 'som_err_map'):
        setattr(diva, attr, None)
    diva.PrepareSimulation()

    # Eagerly create vocal_tract and motor_cortex so gains can be set
    # before the first StartSim call.
    import diva_vocaltract as dvt
    import diva_motorcortex as dmc

    if diva.vocal_tract is None:
        diva.vocal_tract = dvt.VocalTract()
    if diva.motor_cortex is None:
        diva.motor_cortex = dmc.MotorCortex(diva.vocal_tract)


# Normal-condition gains used during the learning phase.
# Per Civier et al. (2010): the feedforward is always trained under normal
# conditions; stuttering represents a change in *production* gains only.
NORMAL_FF = 0.85
NORMAL_FB = 0.30


def make_diva(target_label: str,
              feedforward_gain: float,
              feedback_gain: float,
              learning_phase_runs: int = 4,
              length_override: float = None,
              seed: int = 0):
    """
    Convenience factory: create a trained DIVA instance for a given condition.

    Training always uses normal gains (NORMAL_FF, NORMAL_FB).  After training,
    the gains are switched to the requested test-phase values.  This matches
    Civier et al. (2010): stuttering is a *production* disorder, not a
    learning disorder.
    """
    from main import DIVA

    torch.manual_seed(seed)
    np.random.seed(seed)

    diva = DIVA()
    setup_target(diva, target_label, length_override=length_override)

    # Train with normal gains
    diva.motor_cortex.feedforward_gain = NORMAL_FF
    diva.motor_cortex.feedback_gain = NORMAL_FB

    with torch.no_grad():
        for _ in range(learning_phase_runs):
            diva.StartSim(False, None, False)

    # Switch to test-phase gains
    diva.motor_cortex.feedforward_gain = feedforward_gain
    diva.motor_cortex.feedback_gain = feedback_gain

    # Snapshot post-training state so SelfRepairController can restore it
    # before each test trial.
    #
    # cerebellum.dataBuffer: 1-step delay queue; diva.reset() clears LDA/LDS
    # but NOT dataBuffer, so without restoring it each trial would start with
    # the buffer value left over from the end of the previous trial.
    #
    # LMC weights: AdaptiveWeightBlock.reset() is a no-op; LMC.output() runs
    # a Hebbian update (dW = 0.5*u*v') every step, so without restoring the
    # weights the LMC adapts across trials and errors shrink to zero.
    import copy
    diva._initial_cerebellum_buffer = copy.deepcopy(diva.cerebellum.dataBuffer)
    diva._initial_lmc_weight = diva.motor_cortex.LMC.weight.clone()

    return diva
