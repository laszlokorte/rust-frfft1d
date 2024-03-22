use rustfft::num_complex::Complex;
use rustfft::num_traits::ConstOne;
use rustfft::num_traits::Float;
use rustfft::num_traits::FloatConst;
use rustfft::FftNum;

pub mod interp;

pub fn sinc<T: FftNum + FloatConst + Float>(x: T) -> Complex<T> {
    if x == T::zero() {
        Complex::new(T::one(), T::zero())
    } else {
        Complex::new(T::sin(x * T::PI()) / (x * T::PI()), T::zero())
    }
}
