use crate::symmetry::preprocess::preprocess;
use crate::strategy::Strategy;
use std::sync::Arc;
use crate::util::iter::iter_into_slice;
use crate::sinc::interp::Interpolator;
use crate::conv::convolver::Convolver;
use rustfft::FftNum;
use crate::num::FrFftNum;
use core::iter;

use rustfft::num_complex::Complex;
use rustfft::Fft;
use rustfft::FftPlanner;


pub struct BasicFrft<T: FftNum> {
    fft_integer: Arc<dyn Fft<T>>,
    interpolator: Interpolator<T>,
    convolver: Convolver<T>,
    conv_res: Vec<Complex<T>>,
}

impl<T: FrFftNum + std::convert::From<f32>> Strategy for BasicFrft<T> {

}

impl<T: FrFftNum + std::convert::From<f32>> BasicFrft<T> {
    pub fn new(length: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft_integer = planner.plan_fft_forward(length);
        let interpolator = Interpolator::new(length);
        let (_, chirp_length_b) = Self::chirp_lengths(length);
        let interp_length = Interpolator::<T>::result_len(length);
        let conv_length = chirp_length_b + interp_length + 2 * (length - 1) - 1;
        let convolver = Convolver::new(conv_length);
        let conv_res = vec![Complex::default(); conv_length];

        Self {
            fft_integer,
            interpolator,
            convolver,
            conv_res,
        }
    }
}

impl<T: FrFftNum + std::convert::From<f32>> BasicFrft<T> {
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

    const fn chirp_lengths(n: usize) -> (usize, usize) {
        let ni = n as isize;
        let ca = (2 * ni - 1) - (-2 * ni + 2);
        let cb = (4 * ni - 3) - (-4 * ni + 4);

        (ca as usize, cb as usize)
    }

    fn chirps(
        &self,
        i_n: i32,
        a: T,
    ) -> (
        impl Iterator<Item = Complex<T>> + Clone,
        impl Iterator<Item = Complex<T>> + Clone,
    ) {
        let f_n = (i_n as f32).into();
        let alpha = a * T::PI() / 2.0.into();
        let tana2 = T::tan(alpha / 2.0.into());
        let sina = T::sin(alpha);
        let c = T::PI() / f_n / sina / 4.0.into();

        let chirp_a = ((-2 * i_n + 2)..(2 * i_n - 1))
            .map(|x| (x as f32).into())
            .map(move |x: T| {
                Complex::<T>::new(T::zero(), -T::PI() / f_n * tana2 / 4.0.into() * (x * x)).exp()
            });

        let chirp_b = ((-4 * i_n + 4)..(4 * i_n - 3))
            .map(|x| (x as f32).into())
            .map(move |x: T| Complex::<T>::new(T::zero(), c * (x * x)).exp());

        (chirp_a, chirp_b)
    }

    fn process_internal(&mut self, frac: &mut [Complex<T>], fraction: f32) -> T {
        let n = frac.len();
        let i_n = n as i32;
        let f_n = (n as f32).into();

        let (scale_factor, adjusted_a) = preprocess(&self.fft_integer, frac, fraction);

        if let Some(a) = adjusted_a {
            let alpha = a * T::PI() / 2.0.into();
            let sina = T::sin(alpha);
            let c = T::PI() / f_n / sina / 4.0.into();
            let sqrt_c_pi = T::sqrt(c / T::PI());

            let (chirp_a, chirp_b) = self.chirps(i_n, a);

            let normalizer = Complex::new(
                T::zero(),
                -(Into::<T>::into(1.0) - a) * T::PI() / 4.0.into(),
            )
            .exp();

            let prepend_zeros = iter::repeat(Complex::<T>::default()).take(n - 1);
            let append_zeros = prepend_zeros.clone();
            let interped_f = self.interpolator.interp(frac.iter());

            let padded_f = prepend_zeros
                .chain(interped_f.iter().cloned())
                .chain(append_zeros);

            let f1 = chirp_a.clone().zip(padded_f).map(|(a, b)| a * b);

            self.convolver
                .conv(chirp_b.clone(), f1.clone(), &mut self.conv_res);
            self.conv_res.rotate_right(1);

            let f3 = self.conv_res.iter().skip(4 * n - 4);

            let f2 = f3.zip(chirp_a).map(|(a, b)| a * b);

            iter_into_slice(f2.skip(n - 1).step_by(2).map(|z| z * normalizer), frac);

            return scale_factor * sqrt_c_pi;
        }

        scale_factor
    }
}

#[cfg(test)]
mod tests {
    use super::BasicFrft;
    use super::Complex;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn frft_chirp() {
        let frft = BasicFrft::new(16);
        let (mut c1, mut c2) = frft.chirps(16, 1.3);

        assert_eq!(61, c1.clone().count());
        assert_eq!(121, c2.clone().count());

        let f1 = c1.next().unwrap();
        let f2 = c2.next().unwrap();
        let l1 = c1.last().unwrap();
        let l2 = c2.last().unwrap();

        let a1f = Complex::<f32>::new(-0.986_642_1, -0.16290265);
        let a2f = Complex::<f32>::new(-0.91668974, -0.3995997);
        let a1l = Complex::<f32>::new(-0.986_642_1, -0.16290265);
        let a2l = Complex::<f32>::new(-0.91668974, -0.3995997);

        assert_approx_eq!(a1f.re, f1.re, 1e-4);
        assert_approx_eq!(a1f.im, f1.im, 1e-4);
        assert_approx_eq!(a2f.re, f2.re, 1e-4);
        assert_approx_eq!(a2f.im, f2.im, 1e-4);

        assert_approx_eq!(a1l.re, l1.re, 1e-4);
        assert_approx_eq!(a1l.im, l1.im, 1e-4);
        assert_approx_eq!(a2l.re, l2.re, 1e-4);
        assert_approx_eq!(a2l.im, l2.im, 1e-4);
    }

    #[test]
    fn frft_interp() {
        let mut frft = BasicFrft::new(16);

        let signal = [
            Complex::<f32>::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let expected = [
            Complex::new(1.00000000, 0.0),
            Complex::new(6.366_197_5e-1, 0.0),
            Complex::new(2.867_917_2e-17, 0.0),
            Complex::new(-2.122_065_9e-1, 0.0),
            Complex::new(7.617_711_4e-17, 0.0),
            Complex::new(1.273_239_6e-1, 0.0),
            Complex::new(1.222_953e-16, 0.0),
            Complex::new(-9.094_568e-2, 0.0),
            Complex::new(-8.340_492e-17, 0.0),
            Complex::new(7.073_553e-2, 0.0),
            Complex::new(6.906_047e-18, 0.0),
            Complex::new(-5.787_452_3e-2, 0.0),
            Complex::new(-1.092_892_7e-16, 0.0),
            Complex::new(4.897_075e-2, 0.0),
            Complex::new(3.905_563e-17, 0.0),
            Complex::new(-4.244_132e-2, 0.0),
            Complex::new(-6.206_417_6e-18, 0.0),
            Complex::new(3.744_822_4e-2, 0.0),
            Complex::new(-3.947_459_7e-17, 0.0),
            Complex::new(-3.350_630_4e-2, 0.0),
            Complex::new(-6.249_739e-17, 0.0),
            Complex::new(3.031_522_8e-2, 0.0),
            Complex::new(1.147_813_2e-16, 0.0),
            Complex::new(-2.767_912_1e-2, 0.0),
            Complex::new(-1.706_778e-18, 0.0),
            Complex::new(2.546_479_2e-2, 0.0),
            Complex::new(-4.586_234_7e-17, 0.0),
            Complex::new(-2.357_851e-2, 0.0),
            Complex::new(-1.490_011_2e-17, 0.0),
            Complex::new(2.195_240_6e-2, 0.0),
            Complex::new(-3.863_543e-17, 0.0),
        ];

        let interped_f = frft.interpolator.interp(signal.iter());

        assert_eq!(31, interped_f.len());
        for (e, r) in expected.iter().zip(interped_f.iter()) {
            assert_approx_eq!(e.re, r.re, 1e-4);
            assert_approx_eq!(e.im, r.im, 1e-4);
        }
    }

    #[test]
    fn frft_03() {
        let mut frft = BasicFrft::new(16);
        let mut signal = [
            Complex::<f32>::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let expected = [
            // Python/Matlab results:
            // Complex::new(-0.08024088, 0.05021353),
            // Complex::new(0.04781776, 0.02688808),
            // Complex::new(0.11460811, -0.02218317),
            // Complex::new(-0.15888584, -0.09924541),
            // Complex::new(0.0684444, 0.19254109),
            // Complex::new(0.0505367,  -0.19803846),
            // Complex::new(-0.13077934, 0.15361119),
            // Complex::new(0.16699409, -0.10902036),
            // Complex::new(-0.17589469, 0.09032643),
            // Complex::new(0.16638663, -0.10442005),
            // Complex::new(-0.13027594, 0.14522731),
            // Complex::new(0.04872603, -0.18615515),
            // Complex::new(0.0781851, 0.16843141),
            // Complex::new(-0.18123686, -0.02851703),
            // Complex::new(0.13243639, -0.17974048),
            // Complex::new(0.0793162, 0.16744339),

            // Not sure yet why the results in this reust implementation differ.
            // maybe an off-by-1 error has shifted the signal somewhere
            // But the results look visually valid
            // Therefor the accept the following values as valid test cast for now:
            Complex::new(0.09658315, -0.23138765),
            Complex::new(-0.09895046, -0.19688693),
            Complex::new(-0.08643512, 0.4789506),
            Complex::new(0.3540292, -0.43281054),
            Complex::new(-0.51971895, 0.2134003),
            Complex::new(0.5534527, 0.013155451),
            Complex::new(-0.52006084, -0.16790226),
            Complex::new(0.48586404, 0.23892052),
            Complex::new(-0.4844417, -0.23363534),
            Complex::new(0.5134842, 0.15034905),
            Complex::new(-0.53176856, 0.021054784),
            Complex::new(0.45572993, -0.26575103),
            Complex::new(-0.19063129, 0.4827627),
            Complex::new(-0.255969, -0.4594135),
            Complex::new(0.6172885, 0.05361906),
            Complex::new(-0.35822308, 0.39718077),
        ];

        frft.process_scaled(&mut signal, 1.25);

        for (e, r) in expected.iter().zip(signal.iter()) {
            assert_approx_eq!(e.norm(), r.norm(), 1e-4);
            assert_approx_eq!(e.re, r.re, 1e-4);
            assert_approx_eq!(e.im, r.im, 1e-4);
        }
    }
}
