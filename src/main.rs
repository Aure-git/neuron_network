type Precision = f64;

mod matrix;
use std::{io::Write, path::PathBuf};

use matrix::Matrix;

type DataEntry = (Matrix<Precision>, Matrix<Precision>);
type Data = Vec<DataEntry>;

mod network;

fn main() {
    use network::{Network, read_data, LoadError, SaveError};

    let mut network = match Network::new_from_file() {
        Ok(net) => net,
        Err(LoadError::DecodeFailed) => panic!("The file is corrupted !"),
        Err(LoadError::CouldNotReadDirectory) => panic!("Could not read directory !"),
        _ => {
            Network::new(vec![784, 30, 10])
        }
    };

    let mut stdout = std::io::stdout();

    print!("Reading data...");
    stdout.flush().ok();
    let training_data = read_data(PathBuf::from("./data/training"), 500);
    println!("Half way...");
    let testing_data = read_data(PathBuf::from("./data/testing"), 50);

    println!("Starting learning sequence...");
    network.stochastic_gradient_descent(12, 10, 3.0, 
        training_data, Some(testing_data)
    );

    match network.save() {
        Err(SaveError::CouldNotCreateDirectory) => panic!("Could not create directory './networks' !"),
        Err(SaveError::CouldNotWriteInFile) => panic!("Lacking permission to write in file !"),
        _ => {}
    }

}
