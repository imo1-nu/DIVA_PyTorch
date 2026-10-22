# Fixed transfer-peak states

`transfer_peak_states.json` contains all 40 raw 13-dimensional commands and
native/independent peak references from DIVA-FORWARD-01. This small fixture is
self-contained; running the tests does not require the adjacent research repo.

Source: `imo1-nu/stutter_tts_jp`, branch `research/32-diva-forward-audit`,
`outputs/diva_forward01/results.json` (local, excluded from that repository).
The independent values use the official MATLAB equations transcribed to NumPy,
not an execution of MATLAB. Geometry and H reference code is reproduced in
`experiments/transfer_peak_reference.py` with its origin documented there.
The matching native asset is `src/diva_synth.mat`, SHA256
`cbdb6eab4ba727fa6aa9b4cff3aeb364eab662800c92e6f8b07bfe7104485980`.

The fixture includes W07/W22/W34, current/public trial4/20, zero shape, and all
32 fitted-model centers. Shape/source normalization is fixed; peak ranks are
frequency order, without selecting nearby targets.
