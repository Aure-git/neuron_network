use crate::{matrix::collect::IntoMatrix, Precision, Data};
use crate::Matrix;

use std::fs::read_dir;
use std::path::PathBuf;

use image::{GenericImageView, ImageFormat, ImageReader, Rgba};


#[derive(Debug)]
enum ReadError {
    WrongImagePath,
    WrongFormat,
}
type Result<T> = std::result::Result<T, ReadError>;

/// Grayscale the pixel into a single value between 0 (black) and 1 (white).
fn mean_rgba(rgba : Rgba<u8>) -> Precision {
    ((rgba.0[0] as Precision + rgba.0[1] as Precision + rgba.0[2] as Precision)) / 765.0 // = 3 * 255
}

fn desired_output(digit: usize) -> Matrix<Precision> {
    let mut matrix = Matrix::new_constant((10,1), 0.0);
    matrix.set(1.0, digit, 0).unwrap();

    matrix
}

/// Read a PNG file at `path`, grayscale each pixel and output
/// all values inside a signle column `Matrix`
/// 
/// ## Error
/// If the path leads to nothing or something else than PNG file
fn read_image_as_matrix(path: PathBuf) -> Result<Matrix<Precision>> {
    let mut reader = match ImageReader::open(path) {
        Ok(reader) => reader,
        Err(_) => return Err(ReadError::WrongImagePath)
    };

    reader.set_format(ImageFormat::Png);
    let image = match reader.decode() {
        Ok(image) => image,
        Err(_) => return Err(ReadError::WrongFormat),
    };

    let (width, height) = image.dimensions();
    Ok(image.pixels()
            .map(|(_,_,rgba)| mean_rgba(rgba))
            .into_matrix(((width * height) as usize, 1))
    )
}

fn read_directory(dir_path: PathBuf) -> impl Iterator<Item = Matrix<Precision>> {
    if let Ok(dir) = read_dir(dir_path.clone()) {
        dir.flatten().filter_map(|entry| read_image_as_matrix(entry.path()).ok())
    }
    else {
        panic!("Error : could not find '{:?}'", dir_path);
    }
}


fn read_some_directory(dir_path: PathBuf, cap: u128) -> std::vec::IntoIter<Matrix<Precision>> {
    if let Ok(dir) = read_dir(dir_path.clone()) {
        let mut flat_dir = dir.flatten();
        let mut output = Vec::new();
        let mut count = 0;
        while let Some(entry) = flat_dir.next() {
            match read_image_as_matrix(entry.path()) {
                Ok(matrix) => {
                    output.push(matrix);
                    count += 1;
                    if count >= cap {
                        break;
                    }
                },
                Err(e) => println!("Error while reading image : {e:?}"),
            }
        }

        output.into_iter()
    }
    else {
        panic!("Error : could not find '{:?}'", dir_path);
    }
}

/// ## Parameters
/// `from`: path to _testing_ or _training_ data
/// `cap`: number of tests = 10 * `cap`
/// 
/// ## Note
/// If `cap` is `u128::MAX` then all data will be read
pub fn read_data(from: PathBuf, cap: u128) -> Data {
    let mut data = Vec::new();
    for n in 0..10 {
        if cap == u128::MAX {
            data.extend(read_directory(from.join(n.to_string()))
            .map(|matrix| (matrix, desired_output(n))));
        }
        else {
            data.extend(read_some_directory(from.join(n.to_string()), cap)
            .map(|matrix| (matrix, desired_output(n))));
        }
    }

    data
}