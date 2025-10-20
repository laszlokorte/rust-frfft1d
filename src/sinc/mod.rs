pub use num_traits::ConstOne;
pub use num_traits::Float;
pub use num_traits::FloatConst;
pub use rustfft::num_complex::Complex;
pub use rustfft::FftNum;

pub mod interp;

pub fn sinc<T: FftNum + FloatConst + Float>(x: T) -> Complex<T> {
    if x == T::zero() {
        Complex::new(T::one(), T::zero())
    } else {
        Complex::new(T::sin(x * T::PI()) / (x * T::PI()), T::zero())
    }
}
