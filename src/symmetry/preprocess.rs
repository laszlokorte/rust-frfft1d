use crate::num::FrFftNum;
use rustfft::num_complex::Complex;
use rustfft::Fft;
use std::sync::Arc;

pub fn preprocess<T: FrFftNum + std::convert::From<f32>>(
    fft: &Arc<dyn Fft<T>>,
    frac: &mut [Complex<T>],
    fraction: f32,
) -> (T, Option<T>) {
    let n = frac.len();
    let f_n = n as f32;
    let mut a = (fraction + 4.0).rem_euclid(4.0);

    if a == 0.0 {
        (1.0.into(), None)
    } else if a == 1.0 {
        frac.rotate_right(n / 2);
        fft.process(frac);
        frac.rotate_right(n / 2);

        ((1.0 / f_n).into(), None)
    } else if a == 2.0 {
        frac.reverse();

        (1.0.into(), None)
    } else if a == 3.0 {
        frac.rotate_right(n / 2);
        fft.process(frac);
        frac.rotate_right(n / 2);
        frac.reverse();

        ((1.0 / f_n).into(), None)
    } else {
        let mut scale_factor = 1.0;

        if a > 2.0 {
            frac.reverse();
            a -= 2.0;
        }

        if a > 1.5 {
            a -= 1.0;
            frac.rotate_right(n / 2);
            fft.process(frac);
            frac.rotate_right(n / 2);

            scale_factor /= f_n;
        }
        if a < 0.5 {
            a += 1.0;

            frac.rotate_right(n / 2);
            frac.reverse();
            fft.process(frac);
            frac.rotate_right(n / 2);

            scale_factor *= f_n;
        }

        (scale_factor.into(), Some(a.into()))
    }
}
