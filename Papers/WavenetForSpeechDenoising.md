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
    f(x_t)=sign(x_t)\frac{ln(1+\mu \|x_t\|)}{ln(1+\mu)}
  $$

#### Skip Connections

- Facilitate training deep models
- Enable information at each layer to be propagated directly to the final layers.
- Allows the network to explicitly incorporate features extracted at several hierarchical levels

#### Context Stacks

- Deepen the network without increasing the receptive field length as drastically as increasing the dilation factor does
- Achieved by stacking a set of layers, dilated to some maximum dilation factor on top of each other
- Can be done as many times as required

#### Time Complexity

- Significant drawback of Wavenet is its sequential generation of samples

## Wavenet for Speech Denoising

- Speech denoising techniques aim to improve the intelligibility and overall perceptual quality of speech
- Problem typically formulated as follows
  - $m_t=s_t+b_t$
  - Where
    - $m_t$ is mixed signal
    - $s_t$ is speech signal
    - $b_t$ is background noise
- Goal is to estimate $s_t$ given $m_t$

### Non-Causality

- Some future samples are generally available to help make more well informed predictions
- Model has access to valuable information about samples occuring shortly after a particular sample of interest
- Autoregressive causal nature removed in proposed model.
- Larger filters generally showed inferior performance

### Real-value predictions

- Wavenet uses a discrete softmax output to avoid making any assumption on the shape of the output's distribution
- Suitable for modeling multi-modal distributions
- Early-experimentations with discrete softmax proved disadvantageous
- Real-valued predictions (assuming uni-modal gaussian shaped output distributions) seem to be more appropriate
