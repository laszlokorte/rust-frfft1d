use crate::conv::convolver::Convolver;
use crate::conv::len::conv_length;
use crate::num::FrFftNum;
use crate::sinc::sinc;
use crate::strategy::Strategy;
use crate::symmetry::preprocess::preprocess;
use crate::util::iter::iter_into_slice;
use rustfft::num_complex::Complex;
use rustfft::Fft;
use rustfft::FftNum;
use rustfft::FftPlanner;
use std::sync::Arc;

pub struct FastFrft<T: FftNum> {
    fft_integer: Arc<dyn Fft<T>>,
    convolver: Convolver<T>,

    f1: Vec<Complex<T>>,
    f0c: Vec<Complex<T>>,
    f1c: Vec<Complex<T>>,
    h0: Vec<Complex<T>>,
}

impl<T: FrFftNum + std::convert::From<f32>> Strategy for FastFrft<T> {}

impl<T: FrFftNum + std::convert::From<f32>> FastFrft<T> {
    pub fn new(length: usize) -> Self {
        let sinc_len = 2 * length - 1;
        let fft_conv_len = conv_length(length, sinc_len);

        let mut planner = FftPlanner::new();
        let fft_integer = planner.plan_fft_forward(length);

        Self {
            fft_integer,
            convolver: Convolver::new(fft_conv_len),
            f1: vec![Complex::default(); fft_conv_len],
            f0c: vec![Complex::default(); fft_conv_len],
            f1c: vec![Complex::default(); fft_conv_len],
            h0: vec![Complex::default(); fft_conv_len],
        }
    }

    pub fn process(&mut self, signal: &mut [Complex<T>], fraction: f32) {
        let _ = self.process_internal(signal, fraction);
    }

    pub fn process_scaled(&mut self, signal: &mut [Complex<T>], fraction: f32) {
        let scale = T::sqrt(self.process_internal(signal, fraction));

        for v in signal.iter_mut() {
            v.re *= scale;
            v.im *= scale;
        }
    }

    fn chirps(
        &self,
        n: usize,
        a: T,
    ) -> (
        impl Iterator<Item = Complex<T>> + Clone,
        impl Iterator<Item = Complex<T>> + Clone,
    ) {
        let f_n: T = (n as f32).into();
        let alpha: T = a * T::PI() / 2.0.into();
        let s = T::PI() / (f_n + T::ONE) / alpha.sin() / 4.0.into();
        let t = T::PI() / (f_n + T::ONE) * (alpha / 2.0.into()).tan() / 4.0.into();
        let chirp_a = (0..(2 * n - 1))
            .map(move |x| (x as f32).into())
            .map(move |i: T| -f_n + T::ONE + i)
            .map(move |x: T| Complex::<T>::new(T::ZERO, -T::ONE * t * x * x).exp());
        let chirp_b = (0..(4 * n - 1))
            .map(move |x| (x as f32).into())
            .map(move |i: T| -(Into::<T>::into(2.0) * f_n - T::ONE) + i)
            .map(move |x: T| Complex::<T>::new(T::ZERO, T::ONE * s * x * x).exp());

        (chirp_a, chirp_b)
    }

    fn sinc(&self, n: usize) -> impl Iterator<Item = Complex<T>> {
        let f_n: T = (n as f32).into();

        (0..(2 * n - 1))
            .map(move |x| (x as f32).into())
            .map(move |i: T| Into::<T>::into(2.0) * i)
            .map(move |x: T| x - (Into::<T>::into(2.0) * f_n - Into::<T>::into(3.0)))
            .map(|x: T| (sinc(x) * Into::<T>::into(0.5)))
    }

    fn process_internal(&mut self, frac: &mut [Complex<T>], fraction: f32) -> T {
        let n = frac.len();
        let f_n: T = (n as f32).into();

        let (scale_factor, adjusted_a) = preprocess(&self.fft_integer, frac, fraction);

        if let Some(a) = adjusted_a {
            let alpha = a * T::PI() / 2.0.into();
            let s: T = T::PI() / (f_n + T::ONE) / alpha.sin() / 4.0.into();
            let cs = Complex::<T>::new(T::ZERO, -T::ONE * (T::ONE - a) * T::PI() / 4.0.into())
                .exp()
                / (s / T::PI()).sqrt();

            let (chirp_a, chirp_b) = self.chirps(n, a);

            let sinc_iter = self.sinc(n);

            self.convolver
                .conv(frac.iter().cloned(), sinc_iter, &mut self.f1);
            let f1_slice = self.f1[n..(2 * n - 1)].iter().rev();

            let l0 = chirp_a.clone().step_by(2);
            let l1 = chirp_a.skip(1).step_by(2);
            let e0 = chirp_b.clone().step_by(2);
            let e1 = chirp_b.skip(1).step_by(2);

            let f0m_iter = frac.iter().zip(l0.clone()).map(|(a, b)| a * b);
            let f1m_iter = f1_slice.zip(l1).map(|(a, b)| a * b);

            self.convolver.conv_spectral(f0m_iter, e0, &mut self.f0c);
            self.convolver.conv_spectral(f1m_iter, e1, &mut self.f1c);

            iter_into_slice(
                self.f0c.iter().zip(self.f1c.iter()).map(|(a, b)| a + b),
                &mut self.h0,
            );
            self.convolver.fft(&mut self.h0);
            self.h0.reverse();
            self.h0.rotate_right(1);

            let result = l0
                .enumerate()
                .map(|(i, l)| cs * l * self.h0[n + i] * T::sqrt(f_n));
            iter_into_slice(result, frac);
        }

        scale_factor
    }
}

#[cfg(test)]
mod tests {
    use super::Complex;
    use super::FastFrft;

    #[test]
    fn frft2_0() {
        let mut frft = FastFrft::<f32>::new(4);
        let mut signal = [
            Complex::<f32>::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let expected = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        frft.process_scaled(&mut signal, 0.0);
        assert_eq!(expected, signal);
    }

    #[test]
    fn frft2_1() {
        let mut frft = FastFrft::new(4);
        let mut signal = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let expected = [
            Complex::new(0.5, 0.0),
            Complex::new(-0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(-0.5, 0.0),
        ];

        frft.process_scaled(&mut signal, 1.0);
        assert_eq!(expected, signal);
    }

    #[test]
    fn frft2_2() {
        let mut frft = FastFrft::new(4);
        let mut signal = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let expected = [
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        frft.process_scaled(&mut signal, 2.0);
        assert_eq!(expected, signal);
    }

    #[test]
    fn frft2_3() {
        let mut frft = FastFrft::new(4);
        let mut signal = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let expected = [
            Complex::new(0.5, 0.0),
            Complex::new(-0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(-0.5, 0.0),
        ];

        frft.process_scaled(&mut signal, 3.0);
        assert_eq!(expected, signal);
    }

    #[test]
    fn frft2_4() {
        let mut frft = FastFrft::new(4);
        let mut signal = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let expected = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        frft.process_scaled(&mut signal, 4.0);
        assert_eq!(expected, signal);
    }
}
