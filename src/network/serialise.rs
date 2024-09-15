use super::{Net, Network};

use std::io::{Write, stdin};
use std::fs;
use std::path::PathBuf;

use bitcode::{Decode, Encode};

#[derive(Encode, Decode)]
pub struct NetworkSerial {
    pub biases: Net,
    pub weights: Net
}

#[derive(Debug)]
pub enum SaveError {
    CouldNotCreateDirectory,
    FileAlreadyExist,
    CouldNotWriteInFile,
    NoSaving // The user does not want to save the network
}

#[derive(Debug)]
pub enum LoadError {
    NothingToLoad, // Empty directory or no directory
    CouldNotReadDirectory,
    NotFoundForReading,
    DecodeFailed,
    NoLoading // The user does not want to load something
}

fn write_network(path: PathBuf, network: &NetworkSerial) -> Result<(), SaveError> {
    let mut file = match fs::File::create_new(path) {
        Ok(f) => f,
        Err(_) => return Err(SaveError::FileAlreadyExist),
    };

    let encoded = bitcode::encode(network);
    match file.write_all(&encoded) {
        Err(_) => Err(SaveError::CouldNotWriteInFile),
        Ok(()) => Ok(())
    }
}

fn read_network(path: PathBuf) -> Result<NetworkSerial, LoadError> {
    match fs::read(path) {
        Err(_) => Err(LoadError::NotFoundForReading),
        Ok(data) => {
            match bitcode::decode(&data) {
                Err(_) => Err(LoadError::DecodeFailed),
                Ok(net) => Ok(net)
            }
        }
    }
}

fn get_file_name_for_save() -> String {
    let in_stream = stdin();
    let mut input = String::new();

    println!("Please enter a name for your Network and leave empty to discard: ");
    while let Err(_) = in_stream.read_line(&mut input) {
        continue;
    }

    input.trim().to_owned()
}

fn get_file_name_for_load() -> String {
    let in_stream = stdin();
    let mut input = String::new();

    println!("Please enter the name of your Network and leave empty to skip loadding sequence: ");
    while let Err(_) = in_stream.read_line(&mut input) {
        continue;
    }

    input.trim().to_owned()
}

pub fn save(network: NetworkSerial) -> Result<(), SaveError> {
    let path = PathBuf::from("./networks");
    if !path.exists() && fs::create_dir(&path).is_err() {
        return Err(SaveError::CouldNotCreateDirectory);
    }
    
    
    let mut file_path = path.join(PathBuf::from(get_file_name_for_save()));

    while file_path.exists() {
        if file_path == path {
            return Err(SaveError::NoSaving);
        }
        println!("A file with the same name already exists !");
        file_path = path.join(PathBuf::from(get_file_name_for_save()));
    }

    match write_network(file_path, &network) {
        Err(SaveError::FileAlreadyExist) => {
            save(network) // We assume nobody is crazy enough to reach recursive stack limit by hand
        }
        other => other
    }
}


pub fn load() -> Result<Network, LoadError> {
    let path = PathBuf::from("./networks");
    if !path.is_dir() {
        return Err(LoadError::NothingToLoad); // No directory => no files to load
    }

    if let Ok(mut reader) = fs::read_dir(&path) {
        if reader.next().is_none() {
            return Err(LoadError::NothingToLoad); // No files to load
        }
    }
    else {
        return Err(LoadError::CouldNotReadDirectory);
    }
    

    let mut file_path = path.join(PathBuf::from(get_file_name_for_load()));

    while !file_path.is_file() {
        if file_path == path {
            return Err(LoadError::NoLoading);
        }
        println!("No such file exists in directory !");
        file_path = path.join(PathBuf::from(get_file_name_for_load()));
    }

    read_network(file_path).map(|net| net.into())
}

