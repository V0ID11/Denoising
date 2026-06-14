- In implants enhancing the signal before amplification can significantly reduce discomfort
- Used as a preprocessor stage of speech recognition

- Classic methods include:
  - Spectral Subtraction
  - Wiener filtering
  - Statistical model based methods and subspace algorithms
  - Neural networks have also been used since the 80s

- Denoising auto-encoder architecture has been widely adopted recently
- RNNs also used
- Significant performance exploiting the temporal context information in embedded signals

Most current systems are based on short time Fourier Analysis/synthesis framework

- Recent breakthrough is with GANs
- Generative adversarial networks

- Used really well in computer vision to generate realistic images and generalise welll to pixel-wise, distribuitons

- Main advantages of GANs in Speech enhancment
  - Provides a quick enhancement process. No causality is required and hence there is no recursive opertations like RNNs
  - It works ent to end with the raw audio therefore no hand crafted features
  - Learns from different speakers and noise types
