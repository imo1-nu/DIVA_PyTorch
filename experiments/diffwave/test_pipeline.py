"""Small checks for the added model and loss; no data download or CUDA needed."""
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import ConditionerUpsampler, MelSpectrogram, predict_noise, synthesize
from train import crop_batch, masked_l1
from vendor.model import DiffWave
from vendor.params import AttrDict, params


def tiny_vocoder():
    settings=AttrDict(copy.deepcopy(params))
    settings.residual_layers=2
    settings.residual_channels=4
    settings.dilation_cycle_length=2
    model=DiffWave(settings).eval().requires_grad_(False)
    with torch.no_grad():
        model.output_projection.weight.normal_(0,0.03)
        for _ in range(4): model.diffusion_embedding(torch.tensor([0.0]))
    return model


def test_cached_conditioner_matches_original_forward():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model=tiny_vocoder()
    mel=torch.rand(1,80,5)
    audio=torch.randn(1,5*256)
    step=torch.tensor([10.0])
    with torch.no_grad():
        expected=model(audio,step,mel).squeeze(1)
        actual=predict_noise(model,audio,step,model.spectrogram_upsampler(mel))
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)


def test_conditioner_shape_and_frozen_teacher_gradients():
    torch.set_num_threads(1)
    teacher=tiny_vocoder()
    student=ConditionerUpsampler().train()
    mel=torch.rand(2,80,5)
    with torch.no_grad(): target=teacher.spectrogram_upsampler(mel)
    predicted=student(mel)
    assert predicted.shape==target.shape==(2,80,5*256)
    assert sum(isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)) for m in student.modules())==16
    loss=masked_l1(predicted,target,torch.tensor([1280,1000]))
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())


def test_padding_does_not_affect_loss():
    clean=torch.rand(80,3)
    degraded=torch.rand(80,3)
    x,y,length=crop_batch([(clean,degraded)],[0],np.random.default_rng(1),5)
    assert x.shape==y.shape==(1,80,5)
    assert length.item()==3*256
    reference=torch.ones(1,80,5*256)
    prediction=torch.zeros_like(reference)
    prediction[:,:,3*256:]=10000
    assert masked_l1(prediction,reference,length).item()==1


def test_sampler_seed_and_mel_preprocessing():
    torch.set_num_threads(1)
    model=tiny_vocoder()
    mel=MelSpectrogram()(torch.randn(2048)*0.1)
    assert mel.shape==(80,9)
    assert torch.isfinite(mel).all() and mel.min()>=np.log(1e-5)-1e-6
    assert mel.min()<0
    a=synthesize(model,mel,fast=True,seed=8)
    b=synthesize(model,mel,fast=True,seed=8)
    c=synthesize(model,mel,fast=True,seed=9)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.equal(a,c)
    assert a.shape==(1,9*256)
