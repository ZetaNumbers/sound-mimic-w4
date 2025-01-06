use nalgebra as na;

pub trait Accelerated<T> {
    type Len: na::Dim;
    /// L2 norm over vector
    fn fast_norm(&self) -> T;
    /// Dot product over vectors
    fn fast_dot<S>(&self, rhs: &na::Vector<T, Self::Len, S>) -> T
    where
        S: na::Storage<f32, Self::Len, na::U1> + na::IsContiguous;
    /// L2 metric distance of two vectors
    fn fast_metric_distance<S>(&self, rhs: &na::Vector<T, Self::Len, S>) -> T
    where
        S: na::Storage<f32, Self::Len, na::U1> + na::IsContiguous;

    /// scales vector by `scale` and stores it into `out`
    fn fast_scale<S>(&self, scale: T, dest: &mut na::Vector<T, Self::Len, S>)
    where
        S: na::StorageMut<f32, Self::Len, na::U1> + na::IsContiguous;
}

#[cfg(not(all(feature = "apple-accelerate", target_os = "macos")))]
impl<D, S1> Accelerated<f32> for na::Vector<f32, D, S1>
where
    D: na::Dim,
    S1: na::Storage<f32, D, na::U1> + na::IsContiguous,
{
    type Len = D;

    fn fast_norm(&self) -> f32 {
        self.norm()
    }

    fn fast_dot<S2>(&self, rhs: &nalgebra::Vector<f32, D, S2>) -> f32
    where
        S2: nalgebra::Storage<f32, D, nalgebra::U1> + na::IsContiguous,
    {
        self.dot(rhs)
    }

    fn fast_metric_distance<S2>(&self, rhs: &nalgebra::Vector<f32, D, S2>) -> f32
    where
        S2: nalgebra::Storage<f32, D, nalgebra::U1> + na::IsContiguous,
    {
        self.metric_distance(rhs)
    }

    fn fast_scale<S2>(&self, a: f32, dest: &mut na::Vector<f32, D, S2>)
    where
        S2: nalgebra::StorageMut<f32, D, nalgebra::U1> + na::IsContiguous,
    {
        dest.copy_from(self);
        dest.scale_mut(a);
    }
}

#[cfg(all(feature = "apple-accelerate", target_os = "macos"))]
impl<D, S1> Accelerated<f32> for na::Vector<f32, D, S1>
where
    D: na::Dim,
    S1: na::Storage<f32, D, na::U1> + na::IsContiguous,
{
    type Len = D;

    fn fast_norm(&self) -> f32 {
        let s = self.as_slice();
        apple_accelerate::dot_product(s, s).sqrt()
    }

    fn fast_dot<S2>(&self, rhs: &nalgebra::Vector<f32, D, S2>) -> f32
    where
        S2: nalgebra::Storage<f32, D, nalgebra::U1> + na::IsContiguous,
    {
        let lhs = self.as_slice();
        let rhs = rhs.as_slice();
        apple_accelerate::dot_product(lhs, rhs)
    }

    fn fast_metric_distance<S2>(&self, rhs: &nalgebra::Vector<f32, D, S2>) -> f32
    where
        S2: nalgebra::Storage<f32, D, nalgebra::U1> + na::IsContiguous,
    {
        let lhs = self.as_slice();
        let rhs = rhs.as_slice();
        apple_accelerate::distance_squared(lhs, rhs).sqrt()
    }

    fn fast_scale<S2>(&self, a: f32, dest: &mut na::Vector<f32, D, S2>)
    where
        S2: nalgebra::StorageMut<f32, D, nalgebra::U1> + na::IsContiguous,
    {
        let src = self.as_slice();
        let dest = dest.as_mut_slice();
        apple_accelerate::scale(src, a, dest)
    }
}
