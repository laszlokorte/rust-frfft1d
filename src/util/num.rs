pub(crate) fn next_pow2(n: usize) -> usize {
    2 << f32::ceil(f32::log2(n as f32)) as usize
}
