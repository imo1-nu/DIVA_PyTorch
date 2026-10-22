import torch
import diva_errormap as dem


class AuditoryErrorMap(dem.ErrorMap):

    def __init__(self):
        dem.ErrorMap.__init__(self, 'diva_weights_SSM2amax.mat', 'diva_weights_SSM2amin.mat')
        self.target_max_gain = 1
        self.target_min_gain = 1
        self.error_gain = 1

    def reset(self):
        super().reset()
        self.validity = None

    def output(self):
        validity = getattr(self, 'validity', None)
        if validity is None:
            return super().output()
        original = self.InputPorts[1]
        valid = validity.reshape(-1, 1) & torch.isfinite(original.reshape(-1, 1))
        self.InputPorts[1] = torch.where(valid, original.reshape(-1, 1), 0.)
        try:
            super().output()
            self.OutputPorts[0] = torch.where(valid, self.OutputPorts[0], 0.)
        finally:
            self.InputPorts[1] = original
