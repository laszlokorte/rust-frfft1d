use crate::Complex;
use crate::Float;
use crate::FloatConst;
use rustfft::FftNum;

pub fn sinc<T: FftNum + FloatConst + Float>(x: T) -> Complex<T> {
    if x == T::zero() {
        Complex::new(T::one(), T::zero())
    } else {
        Complex::new(T::sin(x * T::PI()) / (x * T::PI()), T::zero())
    }
}
