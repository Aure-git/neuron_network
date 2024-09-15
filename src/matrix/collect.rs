use super::Matrix;

pub trait IntoMatrix {
    type Item;

    fn into_matrix(self, shape: (usize, usize)) -> Matrix<Self::Item>;
}

impl<I: Iterator> IntoMatrix for I {
    type Item = I::Item;

    /// Create a `Matrix` from any `Iterator`.
    /// 
    /// ## Parameters
    /// shape : (rows, cols)
    /// 
    /// ## Panic
    /// Panics if one of the given size in `shape` is 0 or
    /// if the `Iterator` is empty.
    /// 
    /// Use `cycle` method to create more elements if needed.
    /// If there is too much (or if the iterator already cycle), they will
    /// be lost.
    fn into_matrix(self, shape: (usize, usize)) -> Matrix<Self::Item> {
        Matrix::new_from_iterator(shape, self)
    }
}

pub trait IntoMatrixByCycle {
    type Item;

    fn into_matrix_by_cycle(self, shape: (usize, usize)) -> Matrix<Self::Item>;
}

impl<I: Iterator> IntoMatrixByCycle for I
where I: Sized + Clone {
    type Item = I::Item;

    /// Create a `Matrix` from any `Iterator` by the `cycle` method.
    /// 
    /// ## Parameters
    /// shape : (rows, cols)
    /// 
    /// ## Panic
    /// Panics if the `Iterator` is empty.
    /// 
    /// Also panics if one of the given size in `shape` is 0.
    fn into_matrix_by_cycle(self, shape: (usize, usize)) -> Matrix<Self::Item> {
        Matrix::new_from_cycle(shape, self.cycle())
    }
}