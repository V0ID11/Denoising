- Most recorded speech contains some form of noise that hinder intelligibility
  - e.g. Street noise, dogs barking, keyboard typing etc.
- Particularly important for audio and video calls, hearing aids and other automatic speech recognition systems.
- For many applications a key feature of a speech enhancement system is to run in real time with as little lag as possible (online), on the communication device, preferably on commodity hardware

- Feasible solutions which estimated the noise model and used it to recover noise deducted speech.
- These approaches can generalize well across domains they have trouble dealing with common noises such as non-stationary noise or babble noise from people simultaneously talking
- Presence of this type of noise degrades intelligibility greatly
- DNN based models perform significantly better on non-stationary noise and babble noise whilst generating higher quality speech in objective and subjective evaluations over traditional methods
- Real time version of the DEMUCS architecture adaped for speech enhancement
  - Consists of a causal model
  - Based on convolutions and LSTMs
  - with a frame size of 40 ms,
  - A stride of 16ms
  - Runs faster than real-time on a single laptop CPU core.
  - Goes from wavefrom to waveform through hierarchical generation (using U-Net like skip-connections)
- Model directly outputs clean version of the speech minimizing a regression loss function (L1 loss)
- Enforces real-time constrain on model run-time comparable performance to state of the art model by objective and subjective measures

## Notations and Problem

- Monaural (single-microphone) speech-enhancement can operate in real-time applications.
- Given an audio signal $x\in\mathbb{R}^T$
- Composed of clean speech $y\in\mathbb{R}^T$
- That is corrupted by an additive background signal $n\in\mathbb{R}^T$
- So that $x=y+n$
- The length $T$ is not a fixed value across samples, since the input utterances have different durations.
- Goal is to find an enhancement function $f$ such that $f(x)\approx y$

## DEMUCS architecture

- Multi-layer convolutional encoder and decoder
- With U-Net skip connections
- And a sequence modelling network applied on the encoders' output
- Characterized by its number of layers $L$
- initial number of hidden channels $H$
- Layer kernel size $K$ and stride $S$ and resampling factor $U$
- The encoder and decoder layers are numbered 1 to $L$
- Due to foucsing on the monophonic speech, the input and output of the model has a single channel only
