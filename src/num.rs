pub use num_traits::float::Float;
pub use num_traits::float::FloatConst;
pub use num_traits::identities::ConstOne;
pub use num_traits::identities::ConstZero;
pub use rustfft::FftNum;

pub trait FrFftNum:
    FftNum
    + FloatConst
    + ConstOne
    + ConstZero
    + Float
    + std::default::Default
    + std::ops::RemAssign
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::Mul<Self>
    + std::ops::DivAssign
{
}

impl<
        T: FftNum
            + FloatConst
            + ConstOne
            + Float
            + std::default::Default
            + std::ops::RemAssign
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::Mul<T>
            + std::ops::DivAssign
            + ConstZero,
    > FrFftNum for T
{
}

#[cfg(test)]
mod tests {

    #[test]
    fn frft_chirp() {
        assert_eq!(5.0, is_fft_num(5.0));
        assert_eq!(5.0, is_frfft_num(5.0));
    }

    fn is_frfft_num<T: super::FrFftNum>(n: T) -> T {
        return n;
    }

    fn is_fft_num<T: super::FftNum>(n: T) -> T {
        return n;
    }
}
