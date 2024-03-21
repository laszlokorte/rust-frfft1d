#![feature(iter_array_chunks)]
#![feature(iter_intersperse)]

pub mod convolver;
pub mod frft;
pub mod frft2;
mod iter;
mod sinc;
pub mod sinc_interp;

use crate::convolver::Convolver;
pub use crate::frft::Frft;
pub use crate::frft2::Frft2;
use crate::iter::iter_into_slice;
use rustfft::num_traits::float::Float;
use rustfft::num_traits::float::FloatConst;
use rustfft::num_traits::identities::ConstOne;
use rustfft::num_traits::identities::ConstZero;
use rustfft::Fft;
use rustfft::FftNum;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

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
