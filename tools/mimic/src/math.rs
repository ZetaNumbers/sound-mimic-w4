#[cfg(feature = "cblas")]
extern crate blas_src;

#[cfg(feature = "apple-accelerate")]
mod accelerate {
    /// L2 norm over vector `v`
    #[track_caller]
    pub fn norm(v: &[f32]) -> f32 {
        apple_accelerate::dot_product(v, v)
    }

    /// Dot product over vectors `a` and `b`
    #[track_caller]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        apple_accelerate::dot_product(a, b)
    }

    /// Scale `v` vector by `a` scalar and store into `w` vector
    #[track_caller]
    pub fn scale(v: &[f32], a: f32, w: &mut [f32]) {
        apple_accelerate::scale(v, a, w);
    }

    /// L2 metric distance of two vectors `a`, `b`
    ///
    /// May fill `b` with unspecified data.
    #[track_caller]
    pub fn distance_destructive(a: &[f32], b: &mut [f32]) -> f32 {
        apple_accelerate::distance_squared(a, b).sqrt()
    }
}

#[cfg(feature = "cblas")]
mod cblas {
    /// L2 norm over vector `v`
    #[track_caller]
    pub fn norm(v: &[f32]) -> f32 {
        let n = check_length(v.len());
        unsafe { cblas_sys::cblas_snrm2(n, v.as_ptr(), 1) }
    }

    /// Dot product over vectors `a` and `b`
    #[track_caller]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = check_lengths(a.len(), b.len());
        unsafe { cblas_sys::cblas_sdot(n, a.as_ptr(), 1, b.as_ptr(), 1) }
    }

    /// Scale `v` vector by `a` scalar and store into `w` vector
    #[track_caller]
    pub fn scale(v: &[f32], a: f32, w: &mut [f32]) {
        let n = check_lengths(v.len(), w.len());
        unsafe {
            cblas_sys::cblas_scopy(n, v.as_ptr(), 1, w.as_mut_ptr(), 1);
            cblas_sys::cblas_sscal(n, a, w.as_mut_ptr(), 1);
        }
    }

    /// L2 metric distance of two vectors `a`, `b`
    ///
    /// May fill `b` with unspecified data.
    #[track_caller]
    pub fn distance_destructive(a: &[f32], b: &mut [f32]) -> f32 {
        let n = check_lengths(a.len(), b.len());
        unsafe {
            cblas_sys::cblas_saxpy(n, -1.0, a.as_ptr(), 1, b.as_mut_ptr(), 1);
            cblas_sys::cblas_snrm2(n, b.as_ptr(), 1)
        }
    }

    #[track_caller]
    fn check_lengths(a_len: usize, b_len: usize) -> i32 {
        assert_eq!(a_len, b_len, "input vectors have different lengths");
        check_length(a_len)
    }

    #[track_caller]
    fn check_length(len: usize) -> i32 {
        len.try_into().expect("input data is too large")
    }
}

#[cfg(feature = "apple-accelerate")]
pub use accelerate::*;

// Prioritize apple's framework
#[cfg(all(feature = "cblas", not(feature = "apple-accelerate")))]
pub use cblas::*;
