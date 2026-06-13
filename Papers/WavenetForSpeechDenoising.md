# A Wavenet for Speech Denoising

## Introduction

- Normally not standard practice to work directly in the time domain.
- Most techniques use magnitude spectrograms as front-end
  - Discards potentially valuable information
  - Uses general-purpose feature extractors rather than specific feature representations
- NN handle temporal dependencies effectively
- Most local structure of a speech wave form ($\approx$ tens of milliseconds)
  - In this range many characteristics of the speaker (timbre) can be captured
  - And linguistic patterns in the speech become accessible
  - These levels are not discrete
- Wavenet is an autoregressive generative model
- Uses raw audio over magnitude spectrogram

## Wavenet

- Autoregressive model shapes the probability distribution of the next sample
- Given some fragment of previous samples
- Sequentially feeds previously generated samples back into the model
- Enforces time continuity in the resulting audio waveforms
- Wavenet is an audio domain adaption of the PixelCNN generative model for images

### Key Features

#### Gated Units

- Sigmoidal gates control the activations' contribution in every layer:
- $z_{t'}= tanh(W_f * x_t) \odot \sigma (W_g * x_t)$ where
  - $*$ and $\odot$ operators denote convolution and element wise multiplication respectively
  - $f, t, t' ~ and~g$ stand for filter, input time, output time and gate indices
  - $W_f$ and $W_g$ are convolutional filters

#### Causal, dilated convolutions

- Uses Causal dilated convolutions.
- A series of small (length = 2) convolutional filters with exponentially increasing dilation factors.
- Results in a exponential receptive field growth with depth
- Causality is enforced by asymmetric padding proportional to the dilation factor.
- Prevnets activations propagating back in time.

#### $\mu$-law quantization

- Perform more coarse 8-bit quantization to make the tsk computationally tractable
- Accomplished via a $\mu$-law non-linear companding
  $$
    f(x_t)=sign(x_t)\frac{ln(1+\mu |x_t|)}{ln(1+\mu)}
  $$

## Wavenet for Speech Denoising
