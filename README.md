# Fast Fractional Fourier Transform

This is an implementation of the one dimensional [discrete fractional fourier transform](https://repository.bilkent.edu.tr/server/api/core/bitstreams/5f5ca861-77e1-4a2d-b657-623edb8aafc0/content) (DFrFT) in rust.

## Previews

[A visual preview of results of the discrete fractional fourier transform can be seen here](https://static.laszlokorte.de/frft-cube/).

## References

The implementation is manually translated from the the Matlab code provided by [A.Bultheel, H. Martínez-Sulbaran.](https://nalag.cs.kuleuven.be/research/software/FRFT/) in their paper [Computation of the Fractional Fourier Transform](https://nalag.cs.kuleuven.be/papers/ade/frftcomp/).

They provided two different Matlab implementations [frft.m](https://nalag.cs.kuleuven.be/research/software/FRFT/frft.m) and [frft.m](https://nalag.cs.kuleuven.be/research/software/FRFT/frft2.m).

Both have been translated to rust as `frfft1d::strategy::Basic` and `frfft1d::strategy::Fast` respectively.

## Usage

```rust
use frfft1d::strategy::FastFrft;

use frfft1d::strategy::BasicFrft;

// The Signal to be transformed.
let mut signal = [
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
];

// prepare the calculation for the given signal length.
let mut frft = FastFrft::new(signal.len());

// transform the signal inplace
// the seconds parameter is the fractional exponent of the transform.
// 1.0 corresponds to the classic DFT/FFT.
frft.process_scaled(&mut signal, 1.0);
// 2.0 corresponds to applying the DFT/FFT twice.
frft.process_scaled(&mut signal, 2.0);
// 3.0 corresponds to applying the DFT/FFT three times, which is inturn the same as applying the inverse DFT (iDFT)
frft.process_scaled(&mut signal, 3.0);
// 4.0 corresponds to applying the DFT/FFT four times, which does not change the signal at all.
frft.process_scaled(&mut signal, 4.0);

// The fractional fourier transform does also allow non integer exponents.
// 0.5 corresponds to applying the the DFT only "half".
frft.process_scaled(&mut signal, 0.5);
```
