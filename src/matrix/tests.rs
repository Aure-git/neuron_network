use crate::matrix::Matrix;
use crate::matrix;

use super::collect::IntoMatrixByCycle;

#[test]
fn getter_setter() {
    let mut matrix = Matrix::new_from_iterator((2,2), vec![2,5,-1,3].into_iter());

    assert_eq!(matrix.get(0, 1), Some(&5));
    assert_eq!(matrix.get(3, 1), None);
    assert_eq!(matrix.get(1, 4), None);

    matrix.set(8, 0, 0).unwrap();
    assert_eq!(matrix.get(0, 0), Some(&8));
}

#[test]
fn transpose_matrix() {
    let matrix = Matrix::new_from_iterator((2,4), (0_usize..12).into_iter());
    let transpose = Matrix::new_from_iterator((4,2), vec![0,4,1,5,2,6,3,7].into_iter());
    assert_eq!(matrix.transpose(), transpose);
}

#[test]
fn hadamard() {
    let matrix1 = Matrix::new_from_iterator((2,2), (1_usize..=4).into_iter());
    let matrix2 = Matrix::new_identity((2,2));
    let hadamard = matrix1.hadamard_product(matrix2);

    let expected = Matrix::new_from_iterator((2,2), vec![1,0,0,4].into_iter());
    assert_eq!(hadamard, expected);
}

#[test]
fn map() {
    let matrix = Matrix::new_from_iterator((3,3), (0_usize..9).into_iter());
    let mapped = Matrix::new_from_iterator((3,3), (1..10).into_iter());
    assert_eq!(matrix.map(|x| x + 1), mapped);
}

#[test]
fn addition() {
    let matrix_a = Matrix::new_from_iterator((3,3), (0_usize..9).into_iter().rev());
    let matrix_b = Matrix::new_identity((3,3));

    let sum = Matrix::new_from_iterator((3,3), vec![1,1,2,3,5,5,6,7,9].into_iter().rev());
    assert_eq!(matrix_a + matrix_b, sum);
}

#[test]
fn addition_assign() {
    let mut matrix_a = Matrix::new_from_iterator((3,3), (0_usize..9).into_iter().rev());
    let matrix_b = Matrix::new_identity((3,3));

    matrix_a += matrix_b;

    let sum = Matrix::new_from_iterator((3,3), vec![1,1,2,3,5,5,6,7,9].into_iter().rev());
    assert_eq!(matrix_a, sum);
}

#[test]
fn iter() {
    let matrix: Matrix<usize> = Matrix::new_identity((2,2));
    let mut iterator = matrix.iter();
    
    assert_eq!(iterator.next(), Some(&1));
    assert_eq!(iterator.next(), Some(&0));
    assert_eq!(iterator.next(), Some(&0));
    assert_eq!(iterator.next(), Some(&1));
    assert_eq!(iterator.next(), None);
}

#[test]
fn row_col_iterators() {
    let matrix = Matrix::new_from_iterator((2,3), (0_usize..6).into_iter());
    let mut iter_row = matrix.iter_row(1).unwrap();
    let mut iter_col = matrix.iter_col(2).unwrap();

    assert_eq!(iter_row.next(), Some(&3));
    assert_eq!(iter_row.next(), Some(&4));
    assert_eq!(iter_row.next(), Some(&5));
    assert_eq!(iter_row.next(), None);

    assert_eq!(iter_col.next(), Some(&2));
    assert_eq!(iter_col.next(), Some(&5));
    assert_eq!(iter_col.next(), None);

    assert!(matrix.iter_row(2).is_none());
    assert!(matrix.iter_col(5).is_none());
}

#[test]
fn matrix_product() {
    let matrix_a = Matrix::new_from_iterator((2,3), (1_usize..=6).into_iter());
    let matrix_b = vec![0, 0, 1].into_iter().into_matrix_by_cycle((3,4));

    let output = Matrix::new_from_iterator((2,4), vec![3,2,1,3,6,5,4,6].into_iter());
    assert_eq!(matrix_a * matrix_b, output);
}

#[test]
fn matrix_rule() {
    let matrix_0 = Matrix::new_constant((2,3), -3);
    // let matrix_1 = Matrix::new_from_vec((3,3), vec![1,2,3,1,2,4,1,5,6]);

    assert_eq!(matrix_0, matrix![-3;(2,3)]);
    assert_eq!(matrix_0, matrix![-3;2,3]);
    // assert_eq!(matrix_1, matrix![(3,3); 1,2,3,1,2,4,1,5,6]);
}