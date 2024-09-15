use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use std::fmt;
use std::cmp::min;

pub mod collect;

use collect::{IntoMatrix, IntoMatrixByCycle};
// use serde::{Serialize, Deserialize};
use bitcode::{Decode, Encode};

#[cfg(test)]
mod tests;

/// A matrix over values of the same type
// #[derive(Serialize, Deserialize)]
#[derive(Encode, Decode)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    values: Vec<T>
}

/// Convenience method to compute the number
/// of elements in a `Matrix` from its `shape`
fn size(shape: &(usize, usize)) -> usize {
    shape.0 * shape.1
}

impl<T> Matrix<T> {

    /// Return the _shape_ of the `Matrix`.
    /// 
    /// ## Returned value
    /// (number_of_rows, number_of_columns)
    pub fn get_shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }



    /* Creation */

    /// Create an empty `Matrix`.
    /// 
    /// Because the values in the `Matrix` are __NOT__
    /// set, this function is private and should be use with
    /// caution.
    /// ## Panics
    /// Panics if one of the sizes in `shape` is 0.
    fn new(shape: (usize, usize)) -> Matrix<T> {
        assert!(shape.0 != 0 && shape.1 != 0, "Matrix can't have 0 lenght !");

        Matrix { rows: shape.0, cols: shape.1, values: Vec::with_capacity(size(&shape)) }
    }

    /// Create a `Matrix` from any `Iterator`.
    /// 
    /// ## Parameters
    /// shape : (rows, cols)
    /// 
    /// ## Panic
    /// Panics if the `Iterator` does not have enough elements. (use `new_from_cycle`)
    /// 
    /// If there is more elements than needed, they will be __lost__.
    /// 
    /// Also panics if one of the given sizes in `shape` is 0.
    pub fn new_from_iterator<I>(shape: (usize, usize), iter: I) -> Matrix<T>
    where I: Iterator<Item = T> {
        assert!(shape.0 != 0 && shape.1 != 0, "Matrix can't have 0 lenght !");

        let mut values : Vec<T> = iter.take(size(&shape)).collect();
        assert!(!values.is_empty());

        values.shrink_to_fit();
        
        Matrix { rows: shape.0, cols: shape.1, values }
    }

    /// Create a new `Matrix` with the values of a `Vec`.
    /// 
    /// ## Panics
    /// Panics if the `vec` does not have enough elements to create
    /// the `Matrix`. (use `new_from_cycle`)
    /// 
    /// If there is more elements than needed, they will be __lost__.
    /// 
    /// Also panics if one of the given sizes in `shape` is 0.
    pub fn new_from_vec(shape: (usize, usize), mut vec: Vec<T>) -> Matrix<T> {
        assert!(shape.0 != 0 && shape.1 != 0, "Matrix can't have 0 lenght !");
        assert!(vec.len() >= size(&shape), "{}",
            format_args!("{} elements in Vec but {} are required to create the matrix !", vec.len(), size(&shape))
        );

        vec.truncate(size(&shape));
        vec.shrink_to_fit();

        Matrix { rows: shape.0, cols: shape.1, values: vec }
    }

    /// Create a `Matrix` from a `Cycle`.
    /// 
    /// ## Parameters
    /// shape : (rows, cols)
    /// 
    /// ## Panics
    /// Panics if one of the given sizes in `shape` is 0.
    /// Also panics if the cycle is empty.
    /// 
    /// ## Errors
    /// Return an error if the cycle is empty.
    pub fn new_from_cycle<I>(shape: (usize, usize), cycle: std::iter::Cycle<I>) -> Matrix<T>
    where I: Iterator<Item = T> + Clone + Sized {
        assert!(shape.0 != 0 && shape.1 != 0, "Matrix can't have 0 lenght !");

        let mut values: Vec<T> = cycle.take(size(&shape)).collect(); // Does collect create a well fitting `Vec` ?
        assert!(!values.is_empty());

        values.shrink_to_fit(); // Is this needed ?
        
        Matrix { rows: shape.0, cols: shape.1, values }
    }

    /// Create a new `Matrix` with the same value everywhere.
    /// ## Panics 
    /// Panics if one of the given sizes in `shape` is 0.
    pub fn new_constant(shape: (usize, usize), from : T) -> Matrix<T>
    where T: Sized + Clone + From<u8> {
        assert!(shape.0 != 0 && shape.1 != 0, "Matrix can't have 0 lenght !");
        vec![from].into_iter().into_matrix_by_cycle(shape)
    }

    /// Create a new 'identity' `Matrix` of given any shape.
    /// ## Panics
    /// Panics if one the given sizes in `shape` is 0.
    pub fn new_identity(shape: (usize, usize)) -> Matrix<T> 
    where T: Sized + Clone + From<u8> {
        assert!(shape.0 != 0 && shape.1 != 0, "Matrix can't have 0 lenght !");

        let mut matrix = Matrix::new_constant(shape, T::from(0));
        for i in 0..min(shape.0, shape.1) {
            matrix.values[i*shape.1 + i] = T::from(1);
        }
        matrix
    }

    /// Create a new `Matrix` filled with `default` values.
    /// ## Panics
    /// Panics if one of the given sizes in `shape` is 0.
    pub fn new_default(shape: (usize, usize)) -> Matrix<T>
    where T: Default {
        assert!(shape.0 != 0 && shape.1 != 0, "Matrix can't have 0 lenght !");

        let size = shape.0 * shape.1;
        let mut default = Vec::with_capacity(size);
        (0..size).for_each(|_| default.push(T::default()));
        default.into_iter().into_matrix(shape)
    }



    /* Manipulation */

    /// Tells whether the given coordinates are inside the
    /// `Matrix` or not.
    fn is_in(&self, row: usize, col: usize) -> bool {
        self.rows > row && self.cols > col
    }

    /// Return the element at the given position if it is inside the `Matrix`.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.values.get(row*self.cols + col)
    }

    /// Set the value at the given position.
    /// ## Errors
    /// Return the given value if the position is not in the `Matrix`.
    pub fn set(&mut self, new_value: T, row: usize, col: usize) -> Result<(), T> {
        if !self.is_in(row, col) {
            return Err(new_value);
        }

        self.values[row*self.cols + col] = new_value;
        Ok(())
    }

    /// Return an `Iterator` over the slice.
    /// 
    /// The `Iterator` yields all values of a row before going to the next in order.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.values.iter()
    }

    /// Returns an `Iterator` that allows modifying each value.
    /// 
    /// The `Iterator` yields all values of a row before going to the next in order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.values.iter_mut()
    }

    /// Return an `Iterator` over references of `row`th row.
    fn iter_row(&self, row: usize) -> Option<std::iter::Take<std::iter::Skip<std::slice::Iter<T>>>> {
        if row >= self.rows {
            return None;
        }

        Some(self.values.iter().skip(row * self.cols).take(self.cols))
    }

    /// Return an `Iterator` over the references of `col`th column.
    fn iter_col(&self, col: usize) -> Option<std::iter::StepBy<std::iter::Skip<std::slice::Iter<T>>>> {
        if col >= self.cols {
            return None;
        }

        Some(self.values.iter().skip(col).step_by(self.cols))
    }

    /// Apply the function `f` to each element of the `Matrix`.
    pub fn map<B, F>(self, f: F) -> Matrix<B>
    where F: FnMut(T) -> B {
        let shape = self.get_shape();
        self.into_iter().map(f).into_matrix(shape)
    }

    /// Compute the transpose matrix
    /// 
    /// might not be the fastest...
    pub fn transpose(self) -> Matrix<T> {
        let (new_cols, new_rows) = self.get_shape();
        let mut values = std::collections::VecDeque::from_iter(self.into_iter());

        let mut transpose = Vec::with_capacity(new_rows);
        for _ in 0..new_rows {
            transpose.push(Vec::with_capacity(new_cols));
        }

        for _ in 0..new_cols {
            for old_col in 0..new_rows {
                transpose[old_col].push(values.pop_front().unwrap())
            }
        }

        transpose.into_iter().flatten().into_matrix((new_rows, new_cols))
    }



    /* Computation */

    /// Compute the hadamard product of two matrices.
    pub fn hadamard_product(self, other: Self) -> Matrix<T>
    where T: Mul<Output = T> {
        let shape = self.get_shape();
        self.into_iter().zip(other).map(|(v1, v2)| v1*v2).into_matrix(shape)
    }

    /// Compute the matrix multiplication of two matrices.
    pub fn product(&self, other: Self) -> Matrix<T>
    where T: Mul<T, Output = T> + Sum + Clone {
        self.clone() * other
    }

}

// /// Create a `Matrix` from a vector of vectors of values.
// /// ## Panics
// /// Panics if no shape can be made from the input or if the input is empty.
// fn macro_helper<T>(grid: Vec<Vec<T>>) -> Matrix<T> {
//     let rows = grid.len();
//     let values = grid.into_iter().flatten().collect::<Vec<T>>();
//     let size = values.len();

//     assert!(size % rows == 0);

//     Matrix::new_from_vec((rows, size / rows), values)
// }

/// Create a `Matrix`
#[macro_export]
macro_rules! matrix {
    ($elem:expr; ($r:expr, $c:expr)) => {
        Matrix::new_constant(($r,$c), $elem)
    };
    ($elem:expr; $r:expr, $c:expr) => {
        Matrix::new_constant(($r,$c), $elem)
    };
    // (@shape; $($x:expr),+ $(,)?) => {
    //     crate::matrix::Matrix::new_from_vec(@shape, vec![$($($x),+)+])
    // }
    // (($r:expr, $c:expr); $($x:expr),+ $(,)?) => {
    //     crate::matrix::Matrix::new_from_vec(($r,$c), vec![$($($x),+)+])
    // }
}

impl<T: Clone> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        Matrix { cols: self.cols, rows: self.rows, values: self.values.clone() }
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<U, T: Add<T, Output = U>> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<U>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols, "{}",
            format_args!("Matricies have different shapes : {:?} != {:?}", self.get_shape(), rhs.get_shape())
        );
        let shape = self.get_shape();
        self.into_iter().zip(rhs)
                        .map(|(v1, v2)| v1 + v2)
                        .into_matrix(shape)
    }
}

impl<T: AddAssign> AddAssign<Matrix<T>> for Matrix<T> {
    
    fn add_assign(&mut self, rhs: Matrix<T>) {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols, "{}",
            format_args!("Matricies have different shapes : {:?} != {:?}", self.get_shape(), rhs.get_shape())
        );

        self.iter_mut()
            .zip(rhs)
            .for_each(|(s,r)| *s += r);
    }
}

impl<U, T: Sub<T, Output = U>> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<U>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols, "{}",
            format_args!("Matricies have different shapes : {:?} != {:?}", self.get_shape(), rhs.get_shape())
        );
        let shape = self.get_shape();
        self.into_iter().zip(rhs)
                        .map(|(v1, v2)| v1 - v2)
                        .into_matrix(shape)
    }
}

impl<T: SubAssign> SubAssign<Matrix<T>> for Matrix<T> {
    
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols, "{}",
            format_args!("Matricies have different shapes : {:?} != {:?}", self.get_shape(), rhs.get_shape())
        );

        self.iter_mut()
            .zip(rhs)
            .for_each(|(s,r)| *s -= r);
    }
}

impl<U, T: Mul<T, Output = U> + Clone> Mul<T> for Matrix<T> {
    type Output = Matrix<U>;

    fn mul(self, rhs: T) -> Self::Output {
        let shape = self.get_shape();
        self.into_iter().map(|value| value * rhs.clone()).into_matrix(shape)
    }
}

impl<T: MulAssign + Clone> MulAssign<T> for Matrix<T> {

    fn mul_assign(&mut self, rhs: T) {
        self.iter_mut()
            .for_each(|f| *f *= rhs.clone())
    }
}

// What should be the shape of the returned matrix if the iterator was empty ?
// impl<T: Add<T, Output = T> + Default> Sum for Matrix<T> {

//     fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
//         iter.fold(Matrix::new_default(shape), |acc, m| acc + m)
//     }
// }

impl<U: Sum, T: Mul<T, Output = U> + Clone> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<U>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(self.cols == rhs.rows, "{}",
            format_args!("Matricies have non matching shapes for product ({} != {})", self.cols, rhs.rows)
        );
        let new_shape = (self.rows, rhs.cols);

        let mut vec = Vec::with_capacity(self.rows * rhs.cols);
        for row in 0..new_shape.0 {
            for col in 0..new_shape.1 {
                let iter_row = self.iter_row(row).unwrap();
                let iter_col = rhs.iter_col(col).unwrap();
                vec.push(
                    iter_row.zip(iter_col)
                            .map(|(v1, v2)| v1.clone() * v2.clone())
                            .sum()
                )
            }
        }

        vec.into_iter().into_matrix(new_shape)
    }
}

impl<T: PartialEq> PartialEq for Matrix<T> {
    /// Likely faster than `ne` if both matricies are equal
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.values.iter().zip(other.iter()).all(|(s,o)| s.eq(o))
    }


    /// Override for faster computation (if both matricies are equal)
    /// 
    /// Likely faster than `eq` if both matricies are equal
    fn ne(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return true;
        }
        self.values.iter().zip(other.iter()).any(|(v1, v2)| v1.ne(v2))
    }
}

impl<T: fmt::Debug> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for y in 0..(self.rows-1) {
            write!(f, "[")?;
            for x in 0..(self.cols-1) {
                write!(f, "{:?}, ", self.values[y* self.cols + x])?;
            }
            write!(f, "{:?}]\n", self.values[(y+1)*self.cols - 1])?;
        }

        write!(f, "[")?;
        for x in 0..(self.cols-1) {
            write!(f, "{:?}, ", self.values[(self.rows-1)*self.cols + x])?;
        }
        write!(f, "{:?}]", self.values[self.cols*self.rows -1])?;

        write!(f, "]")
    }
}