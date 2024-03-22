use rustfft::num_traits::float::Float;
use rustfft::num_traits::float::FloatConst;
use rustfft::num_traits::identities::ConstOne;
use rustfft::num_traits::identities::ConstZero;
use rustfft::FftNum;

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
            + rustfft::num_traits::ConstZero,
    > FrFftNum for T
{
}
