---
language: "en"
inference: false
tags:
- Vocoder
- DiffWave
- text-to-speech
- TTS
- speech-synthesis
- speechbrain
license: "apache-2.0"
datasets:
- LJSpeech
---

# Vocoder with DiffWave trained on LJSpeech

This repository provides all the necessary tools for using a [DiffWave](https://arxiv.org/pdf/2009.09761.pdf) vocoder trained with [LJSpeech](https://keithito.com/LJ-Speech-Dataset/). 

The pre-trained model takes as input a spectrogram and generates a waveform as output. Typically, a vocoder is used after a TTS model that converts an input text into a spectrogram.

The sampling frequency is 22050 Hz.


## Install SpeechBrain

```bash
pip install speechbrain
```

Please notice that we encourage you to read our tutorials and learn more about
[SpeechBrain](https://speechbrain.github.io).

### Using the Vocoder as reconstructor
```python
import torch
import torchaudio
import speechbrain as sb
from speechbrain.inference.vocoders import DiffWaveVocoder
from speechbrain.lobes.models.HifiGAN import mel_spectogram
from speechbrain.utils.fetching import fetch
from speechbrain.utils.data_utils import split_path

diffwave = DiffWaveVocoder.from_hparams(source="speechbrain/tts-diffwave-ljspeech", savedir="pretrained_models/tts-diffwave-ljspeech")

source, fl = split_path("speechbrain/tts-diffwave-ljspeech/LJ050-0075.wav")
path = fetch(fl, source=source, savedir="tmpdir")
audio, fs_file = torchaudio.load(path)
audio = torch.FloatTensor(audio)

mel = mel_spectogram(
    sample_rate=22050,
    hop_length=256,
    win_length=1024,
    n_fft=1024,
    n_mels=80,
    f_min=0,
    f_max=8000,
    power=1.0,
    normalized=False,
    norm="slaney",
    mel_scale="slaney",
    compression=True,
    audio=audio,
)

# Running Vocoder (spectrogram-to-waveform), a fast sampling can be realized by passing user-defined variance schedules. According to the paper, high-quality audios can be generated with only 6 steps (instead of a total of 50).
waveforms = diffwave.decode_batch(
    mel,
    hop_len=256,  # upsample factor, should be the same as "hop_len" during the extraction of mel-spectrogram
    fast_sampling=True,  # fast sampling is highly recommanded
    fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],  # customized noise schedule 
)

torchaudio.save('reconstructed.wav', waveforms.squeeze(1), 22050)
```

### Using the Vocoder with TTS
```python
import torchaudio
from speechbrain.pretrained import FastSpeech2
from speechbrain.pretrained import DiffWaveVocoder

# Intialize TTS (FastSpeech2) and Vocoder (DiffWave)
fastspeech2 = FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir="pretrained_models/tts-fastspeech2-ljspeech")
diffwave = DiffWaveVocoder.from_hparams(source="speechbrain/tts-diffwave-ljspeech", savedir="pretrained_models/tts-diffwave-ljspeech")

input_text = "This is a test run with FastSpeech and DiffWave."

# Running the TTS
mel_output, durations, pitch, energy = fastspeech2.encode_text(
  [input_text],
  pace=1.0,        # scale up/down the speed
  pitch_rate=1.0,  # scale up/down the pitch
  energy_rate=1.0, # scale up/down the energy
)

# Running Vocoder (spectrogram-to-waveform), a fast sampling can be realized by passing user-defined variance schedules. According to the paper, high-quality audios can be generated with only 6 steps (instead of a total of 50).
waveforms = diffwave.decode_batch(
    mel_output,
    hop_len=256,  # upsample factor, should be the same as "hop_len" during the extraction of mel-spectrogram
    fast_sampling=True,  # fast sampling is highly recommanded
    fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],  # customized noise schedule 
)

# Save the waverform
torchaudio.save('example_TTS.wav',waveforms.squeeze(1), 22050)
```

### Inference on GPU
To perform inference on the GPU, add  `run_opts={"device":"cuda"}`  when calling the `from_hparams` method.

### Training
The model was trained with SpeechBrain.
To train it from scratch follow these steps:
1. Clone SpeechBrain:
```bash
git clone https://github.com/speechbrain/speechbrain/
```
2. Install it:
```bash
cd speechbrain
pip install -r requirements.txt
pip install -e .
```
3. Run Training:
```bash
cd recipes/LJSpeech/TTS/vocoder/diffwave/
python train.py hparams/train.yaml --data_folder /path/to/LJspeech
```
You can find our training results (models, logs, etc) [here](https://www.dropbox.com/sh/tbhpn1xirtaix68/AACvYaVDiUGAKURf2o-fvgMoa?dl=0).


### Limitations
The SpeechBrain team does not provide any warranty on the performance achieved by this model when used on other datasets.

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```