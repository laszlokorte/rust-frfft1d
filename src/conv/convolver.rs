use crate::num::FrFftNum;
use crate::util::iter::iter_into_slice;
use rustfft::Fft;
use rustfft::FftNum;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

pub struct Convolver<T: FftNum> {
    fft_conv: Arc<dyn Fft<T>>,
    pad_a: Vec<Complex<T>>,
    pad_b: Vec<Complex<T>>,
}

impl<T: FftNum + Default> Convolver<T> {
    pub fn new(length: usize) -> Self {
        let mut pad_a = vec![Complex::default(); length];
        let mut pad_b = vec![Complex::default(); length];
        let mut planner = FftPlanner::new();
        let fft_conv = planner.plan_fft_forward(length);

        fft_conv.process(&mut pad_a);
        fft_conv.process(&mut pad_b);

        Self {
            fft_conv,
            pad_a,
            pad_b,
        }
    }
}

impl<T: FrFftNum + std::convert::From<f32>> Convolver<T> {
    pub fn conv_spectral(
        &mut self,
        a: impl Iterator<Item = Complex<T>>,
        b: impl Iterator<Item = Complex<T>>,
        into: &mut [Complex<T>],
    ) {
        self.pad_a.fill(Complex::default());
        self.pad_b.fill(Complex::default());

        iter_into_slice(a, &mut self.pad_a);
        iter_into_slice(b, &mut self.pad_b);

        self.fft_conv.process(&mut self.pad_a);
        self.fft_conv.process(&mut self.pad_b);

        iter_into_slice(
            self.pad_a.iter().zip(self.pad_b.iter()).map(|(a, b)| a * b),
            into,
        )
    }

    pub fn conv(
        &mut self,
        a: impl Iterator<Item = Complex<T>>,
        b: impl Iterator<Item = Complex<T>>,
        into: &mut [Complex<T>],
    ) {
        self.conv_spectral(a, b, into);
        self.fft_conv.process(into);
        into.reverse();
        let scale: T = (self.pad_a.len() as f32).into();

        for r in into.iter_mut() {
            *r /= scale;
        }
    }

    pub fn fft(&self, signal: &mut [Complex<T>]) {
        self.fft_conv.process(signal);
    }
}
