# MRT2 Temporal Body Streaming Carry Validation

- Steps: 64 (window: 41; wrapped: 23)
- Boundary: 48 ordinary cache tensors in, 48 one-frame updates out
- PyTorch streaming vs reference correlation: 0.999999996159
- PyTorch fresh vs warmed diverged: True
- Core ML vs reference correlation: 0.887406652154
- Core ML vs reference max error: 23.9488573074
- Core ML finite ratio: 1.000000
- Core ML fresh vs warmed diverged: True

This receipt crosses the 41-frame window and therefore exercises host-ring wraparound, not only no-wrap startup.
