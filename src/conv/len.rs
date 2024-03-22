use crate::util::num::next_pow2;

pub fn conv_length(a_size: usize, b_size: usize) -> usize {
    let n = a_size + b_size - 1;

    next_pow2(n)
}