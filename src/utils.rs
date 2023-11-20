pub trait Lerp: Sized {
    fn lerp(self, a: Self, b: Self) -> Self;

    fn unlerp(self, a: Self, b: Self) -> Self;
}

impl Lerp for f32 {
    fn lerp(self, a: f32, b: f32) -> f32 {
        self.mul_add(b - a, a)
    }

    fn unlerp(self, a: f32, b: f32) -> f32 {
        (self - a) / (b - a)
    }
}

impl Lerp for f64 {
    fn lerp(self, a: f64, b: f64) -> f64 {
        self.mul_add(b - a, a)
    }

    fn unlerp(self, a: f64, b: f64) -> f64 {
        (self - a) / (b - a)
    }
}
