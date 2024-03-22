use crate::conv::convolver::Convolver;
use crate::num::FrFftNum;
use crate::sinc::sinc;
use crate::sinc::Complex;
use crate::sinc::ConstOne;
use crate::sinc::Float;
use crate::sinc::FloatConst;
use itertools::intersperse;
use rustfft::FftNum;

pub struct Interpolator<T: FftNum> {
    len: usize,
    convolver: Convolver<T>,
    conv_result: Vec<Complex<T>>,
}

impl<
        T: FftNum + FloatConst + Float + std::default::Default + std::convert::From<f32> + ConstOne,
    > Interpolator<T>
{
    const fn conv_length(length: usize) -> usize {
        (length * 3 - 1) + (3 * length - 5) - 1
    }

    fn sinc_iter(length: isize) -> impl Iterator<Item = Complex<T>> {
        let two = T::ONE + T::ONE;
        ((-2 * length + 3)..(2 * length - 2)).map(move |x| sinc(Into::<T>::into(x as f32) / two))
    }

    const fn slice_range(length: usize) -> std::ops::Range<usize> {
        (2 * length - 4)..(Self::conv_length(length) - 2 * length + 2)
    }

    pub const fn result_len(length: usize) -> usize {
        let r = Self::slice_range(length);

        r.end - r.start
    }

    pub fn new(length: usize) -> Self {
        Self {
            len: length,
            convolver: Convolver::new(Interpolator::<T>::conv_length(length)),
            conv_result: vec![Complex::default(); Interpolator::<T>::conv_length(length)],
        }
    }
}

impl<T: FrFftNum + std::convert::From<f32>> Interpolator<T> {
    pub fn interp<'s, 'c>(
        &'s mut self,
        signal: impl Iterator<Item = &'c Complex<T>> + Clone,
    ) -> &'s [Complex<T>] {
        let interspersed = intersperse(signal.clone().cloned(), Complex::default());

        self.convolver.conv(
            interspersed,
            Self::sinc_iter(self.len as isize),
            &mut self.conv_result,
        );

        &self.conv_result[Self::slice_range(self.len)]
    }
}

#[cfg(test)]
mod tests {

    use crate::sinc::interp::Interpolator;
    use crate::sinc::Complex;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_interp() {
        let signal = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let mut interpolator = Interpolator::<f32>::new(3);

        let result = interpolator.interp(signal.iter());
        let expected = [
            Complex::new(1., 0.0),
            Complex::new(1.273_239_5, 0.0),
            Complex::new(2., 0.0),
            Complex::new(2.970_892_2, 0.0),
            Complex::new(3., 0.0),
        ];

        assert_eq!(5, result.len());
        for (e, r) in expected.iter().zip(result.iter()) {
            assert_approx_eq!(e.re, r.re, 1e-4);
            assert_approx_eq!(e.im, r.im, 1e-4);
        }
    }
}
