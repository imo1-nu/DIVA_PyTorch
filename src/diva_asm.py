import torch
import diva_delayblock as db


class AuditoryStateMap(db.DelayBlock):
    """Keep observation validity and provenance on the existing 10-step delay."""
    def __init__(self, masked=False):
        self.masked = masked
        db.DelayBlock.__init__(self, 10)
        self.set_inputs(1)
        self.set_outputs(1)
        self.scalar_gain = 1
        self._reset_metadata()

    def _reset_metadata(self):
        initial = torch.ones((self.InputPortDimensions[0], 1), dtype=torch.bool)
        if self.masked:
            initial[1:] = False
        self.metadata_buffer = [(initial.clone(), None) for _ in range(self.delay_steps)]
        self.validity = initial.clone()
        self.observation = None

    def setinputdims(self, dm):
        db.DelayBlock.setinputdims(self, dm)
        self._reset_metadata()

    def input(self, data, observation=None):
        db.DelayBlock.input(self, data)
        valid = torch.ones_like(data, dtype=torch.bool) if observation is None else observation.valid.clone()
        self.metadata_buffer.append((valid, observation))

    def output(self):
        db.DelayBlock.output(self)
        if self.metadata_buffer:
            self.validity, self.observation = self.metadata_buffer.pop(0)
        else:
            self._reset_metadata()

    def reset(self):
        db.DelayBlock.reset(self)
        self._reset_metadata()
