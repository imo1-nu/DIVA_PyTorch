# Vendored DiffWave architecture

`model.py` and `params.py` are copies of LMNT's DiffWave architecture and
configuration, retrieved on 2026-10-23 from:

- https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/model.py
- https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/params.py

Copyright 2020 LMNT, Inc. Apache License 2.0; see [LICENSE](LICENSE).
These two files retain their original copyright headers and were not edited.
Their SHA256 values are recorded in `../results/source_provenance.json`.

`../model.py` provides independent checkpoint-name conversion, explicitly sets
SpeechBrain reflection padding, computes cached conditioning during inference,
and adds the experiment's independently trained CNN. SpeechBrain checkpoint
weights are downloaded separately (Apache 2.0 per its model card), with a fixed
revision and SHA256; they are not included in this repository.

The CNN is an explicitly documented interpretation of Zhang et al. (2021),
not recovered author code. See `../README.md` for the 15/16-layer ambiguity.
