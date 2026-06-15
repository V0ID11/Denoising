- Instead of explicitly modelling the extra noise
- We focus on learning a mapping between nosy speech spectra and clean speech spectra
- Model size often exceeds several hundreds of megabytes limiting usability on embedded systems
- CNNs consist of fewer parameters than FNNs and RNNs due to its weight sharing property
- CNNs already proved efficacy on extracting features in speech recognition
  - or on eliminating noises in images
- Can perform better than other NN styles with a much smaller network size
- Uses a new architecture
  - Redundant Convolutional Encoder Decoder (R-CED)
  - Extracts redundant representations of a noisy spectrum at the encoder
  - Maps it back to clean a spectrum at the decoder
  - Maps the spectrum to higher dimensions and projects the features back to lower dimensions

## The problem

- Given a noisy spectra $\{x_t\}^T_{t=1}$ and clean spectra $\{y_t\}^T_{t=1}$, our aim is to learn a mapping $f$ which generates a segment of 'denoised' spectra $\{f(x_t)\}^T_{t=1}$
- That approximates the clean spectra in the $l_2$ norm
  $$
    min~\sum_{t=1}^{T} ||y_t - f(x_t)||^2_2
  $$
- We formulate $f$ using a neural network
