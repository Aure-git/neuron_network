mod serialise;
use serialise::NetworkSerial;


/// Read **PNG** files.
/// 
/// Every PNG file, grayscale each pixel create a `Matrix` from it.
/// Also create the desired output from that PNG file according to
/// the name of the directory the file is in.
pub mod reader;

/// Convenient export for user
pub use reader::read_data;
pub use serialise::{SaveError, LoadError};

use crate::{Data, DataEntry, Matrix, Precision};
use crate::matrix::collect::IntoMatrix;

use std::io::Write;

use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};
use rand_distr::StandardNormal;
// use serde::de::Visitor;
// use serde::ser::SerializeStruct;
// use serde::{Deserialize, Serialize};

type Layer = Matrix<Precision>;
type Net = Vec<Layer>;

struct TestUnit {
    best_score: usize,
    quantity: usize,
    best_weights: Net,
    best_biases: Net
}

/// A neural network
pub struct Network {
    length: usize,
    biases: Net,
    weights: Net,
    rng: ThreadRng,
    test_unit: Option<TestUnit>,
}

/// Compute the sigmoïd function defined as
/// 
/// sig(x) = 1 / (1 + exp(-x))
fn sigmoide(float: Precision) -> Precision {
    1.0 / (1.0 + (-float).exp())
}

/// Compute the derivative of the sigmoïd function
/// 
/// sig'(x) = sig(x) * (1 - sig(x))
fn sigmoide_derivate(float: Precision) -> Precision {
    let sig = sigmoide(float);
    sig * (1.0 - sig)
}

/// Return the percentage of score
/// 
/// ## Parameters
/// positive : number of positive occurence
/// total : number of tests
fn compute_score(positive: usize, total: usize) -> f32 {
    (10000.0 * (positive as f32) / (total as f32)).round() / 100.0
}

/// Found the index of the largest number in a COLUMN `Matrix`
fn key_of_max(matrix: &Matrix<Precision>) -> usize {
    matrix.iter().enumerate().max_by(|(_,a),(_,b)| a.total_cmp(b)).unwrap().0
}

/// Use `StandartNormal` distribution (Gaussian distribution with
/// mean 0 and variance 1) to create a `Matrix` with random values
fn new_gaussian_matrix(rng: &mut ThreadRng, shape: (usize, usize)) -> Matrix<Precision> {
    rng.sample_iter(StandardNormal).take(shape.0 * shape.1).into_matrix(shape)
}

impl Network {

    /// Create a new neural `Network` with given `shape`.
    /// 
    /// Use `StandartNormal` distribution (Gaussian distribution with
    /// mean 0 and variance 1) to initialise the neurons.
    /// 
    /// ##Parameters
    /// shape: from left to right, number of neuron in each layer
    /// 
    /// First layer should have as many neuron as input values (784)
    /// and last neuron as many as possible outputs (10)
    pub fn new(shape: Vec<usize>) -> Self {
        let length = shape.len();
        let mut biases = Vec::with_capacity(length);
        let mut weights = Vec::with_capacity(length);

        let mut rng = rand::thread_rng();

        shape.iter().skip(1).for_each(|&size| biases.push(
            new_gaussian_matrix(&mut rng, (size,1))
        ));

        shape.iter().skip(1).zip(shape.iter()).for_each(|(&rows, &cols)| weights.push(
            new_gaussian_matrix(&mut rng, (rows, cols))
        ));
        
        Network { length, biases, weights, rng, test_unit: None }
    }

    /// Initialises the `test_unit` with the length of the `test_data`
    fn init_test_unit(&mut self, quantity: usize) {
        self.test_unit = Some(TestUnit {
            best_score: 0,
            quantity,
            best_weights: Vec::with_capacity(self.length),
            best_biases: Vec::with_capacity(self.length)
        });
    }

    /// Return the length of the testing data
    /// 
    /// ## Panics
    /// Panics if there is no testing data or if
    /// the `test_unit` if NOT initialised
    fn testing_size(&self) -> usize {
        if let Some(test_unit) = &self.test_unit {
            return test_unit.quantity;
        }
        unreachable!();
    }

    /// Update the best score
    fn check_update_score(&mut self, score: usize) {
        if let Some(test_unit) = &mut self.test_unit {
            if score > test_unit.best_score {
                test_unit.best_score = score;
                test_unit.best_weights = self.weights.clone();
                test_unit.best_biases = self.biases.clone();
            }
        }
    }

    /// Compute the output of the `Network` from a given input `matrix`
    fn feedforward(&self, mut matrix: Matrix<Precision>) -> Matrix<Precision> {
        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            matrix = (weight.clone() * matrix + bias.clone()).map(|value| sigmoide(value))
        };

        matrix
    }

    /// The backpropagation algoritm take a `Matrix` as input,
    /// pass it throughout the `Network` and compute the error
    /// for both the weights and the biases
    /// 
    /// ## Parameters
    /// entry = (input, expected_output)
    /// ## Output
    /// (weights_error, biases_error)
    fn backpropagation(&self, entry: DataEntry) -> (Vec<Matrix<Precision>>,Vec<Matrix<Precision>>) {
        let (start, goal) = entry;
        let mut inputs = Vec::with_capacity(self.length +1);
        let mut outputs = Vec::with_capacity(self.length);

        inputs.push(start);

        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            let input = inputs.last().unwrap();
            let ouput = weight.clone() * input.clone() + bias.clone();
            outputs.push(ouput.clone());
            inputs.push(ouput.map(|value| sigmoide(value)));
        }

        let mut delta = (inputs.pop().unwrap() - goal.clone())
            .hadamard_product(outputs.pop().unwrap().map(|value| sigmoide_derivate(value)));

        let mut biases_error = vec![delta.clone()];
        let mut weights_error = vec![delta.clone() * inputs.pop().unwrap().transpose()];

        for i in (1..self.length-1).rev() {
            delta = (self.weights.get(i).unwrap().clone().transpose() * delta)
                .hadamard_product(
                    outputs.pop().unwrap().map(|value| sigmoide_derivate(value))
                );
            biases_error.push(delta.clone());
            weights_error.push(delta.clone() * inputs.pop().unwrap().transpose());
        }

        weights_error.reverse();
        biases_error.reverse();
        (weights_error, biases_error)
    }

    /// Take some `Data` in a `batch` and learn from it using the
    /// `backpropagation` algorithm.
    /// 
    /// Sums the error of all entry in the `batch` before mutating
    /// the weights and the biases of the `Network`.
    /// 
    /// ## Panics
    /// Batch should NOT be empty
    fn train_with_batch(&mut self, batch: Data, learning_rate: Precision) {
        // let mut gradient_biases = Vec::with_capacity(self.length);
        // let mut gradient_weights = Vec::with_capacity(self.length);

        // self.biases.iter().for_each(|bias| 
        //     gradient_biases.push(Matrix::new_constant(bias.get_shape(), 0.0))
        // );
        // self.weights.iter().for_each(|weight|
        //     gradient_weights.push(Matrix::new_constant(weight.get_shape(), 0.0))
        // );

        // for entry in batch {
        //     let (delta_weights, delta_biases) = self.backpropagation(entry);

        //     gradient_biases.iter_mut()
        //         .zip(delta_biases)
        //         .for_each(|(sum, delta)| *sum += delta);

        //     gradient_weights.iter_mut()
        //         .zip(delta_weights)
        //         .for_each(|(sum, delta)| *sum += delta);
        // }

        // Ah ah ! Rust goes BRRRRRR for compactness

        let coefficient = -learning_rate / (batch.len() as Precision);

        let (gradient_weights, gradient_biases) = batch.into_iter()
            .map(|entry| self.backpropagation(entry))
            .reduce(|(sum_w, sum_b),(delta_w, delta_b)| (
                sum_w.into_iter().zip(delta_w).map(|(s,dw)| s + dw).collect(),
                sum_b.into_iter().zip(delta_b).map(|(s,db)| s + db).collect()
            )).unwrap(); // Batch is never empty, we can unwrap
        
        
        self.weights.iter_mut()
                    .zip(gradient_weights)
                    .for_each(|(current, error)| *current += error * coefficient);
        self.biases.iter_mut()
                    .zip(gradient_biases)
                    .for_each(|(current, error)| *current += error * coefficient);
    }

    /// Test the `Network` with the given `testing_data`
    /// and return how many correct guesses it made.
    /// 
    /// ## Overflow
    /// Can overflow if `testing_data` has over `USIZE::MAX` elements 
    fn evaluate(&self, testing_data: &Data) -> usize {
        let mut score = 0;
        for (input, goal) in testing_data {
            if key_of_max(&self.feedforward(input.clone())) == key_of_max(goal) {
                score += 1;
            }
        }

        score
    }

    /// Train the `Network` using a Stochastic Gradient Descent method.
    /// 
    /// ## Hyper Parameters
    /// `batch_size` : number of training images before a computing the error and
    /// mutating the `Network`
    /// 
    /// `learning_rate` : coefficient of mutation. Train slower with too small values
    /// and learn bad with too big values
    /// 
    /// ## Note on parameters
    /// `testing` : if `None` is given, no score will be shown. It should not be the
    /// same data as the `training` data for a better test.
    /// 
    /// ## Panics
    /// The `training` data should __NOT__ be empty ! (same for the _testing_ data
    /// if some was given)
    pub fn stochastic_gradient_descent(&mut self, generation: u64, batch_size: usize,
        learning_rate: Precision, mut training: Data, testing: Option<Data>) {
        
        let mut stdout = std::io::stdout();
        let training_size = training.len();
        if let Some(test_data) = &testing {
            self.init_test_unit(test_data.len());
            print!("Evalution before training: ");
            stdout.flush().ok();
            let score = self.evaluate(test_data);
            self.check_update_score(score);
            let testing_size = self.testing_size();
            println!("{score} / {} = {}%", testing_size, compute_score(score, testing_size));
        }
        
        for gen in 1..=generation {
            print!("Gen {gen}: 0%... ");
            stdout.flush().ok();
            training.shuffle(&mut self.rng);

            (0..training_size)
                .step_by(batch_size)
                .inspect(|&k| {
                    let x = (100.0 * (k as f32) / (training_size as f32)) / 25.0;
                    if (x-1.0).abs() < f32::EPSILON || (x-2.0).abs() < f32::EPSILON || (x-3.0).abs() < f32::EPSILON {
                        print!("{}%... ", x*25.0);
                        stdout.flush().ok();
                    }

                })
                .for_each(|k| self.train_with_batch(
                    training.get(k..(k + batch_size)).unwrap().to_owned(), learning_rate
                ));
            
            println!("100% !");

            if let Some(testing_data) = &testing {
                print!("Evaluation sequence: ");
                stdout.flush().ok();
                let score = self.evaluate(testing_data);
                self.check_update_score(score);
                let testing_size = self.testing_size();
                println!("{score} / {} = {}%", testing_size, compute_score(score, testing_size));
            }
        }
        
    }

    /// Try to save the `Network` as a file.
    /// 
    /// If the no test data was given to the network, nothing
    /// will happen. But `Ok` will be returned anyway.
    /// 
    /// ## Errors
    /// + Lacking permission to create a directory
    /// + Lacking permission to access, read and write in the directory
    pub fn save(self) -> Result<(), SaveError> {
        if self.test_unit.is_some() {
            return serialise::save(self.into());
        }

        Ok(())
    }

    /// Try to load a `Network` from the `./networks` directory
    /// 
    /// ## Errors
    /// + User does not want to load a network
    /// + Lacking permission to access and read in the directory or the files
    /// + Directory is empty or does not exist, thus nothing can be loaded
    /// + Corrupted data
    pub fn new_from_file() -> Result<Self, LoadError> {
        serialise::load()
    }
}

impl From<NetworkSerial> for Network {
    fn from(value: NetworkSerial) -> Self {
        let length = value.biases.len() +1;
        Self { length, biases: value.biases, weights: value.weights, rng: rand::thread_rng(), test_unit: None }
    }
}

impl Into<NetworkSerial> for Network {
    fn into(self) -> NetworkSerial {
        if let Some(test_unit) = self.test_unit {
            return NetworkSerial {biases: test_unit.best_biases, weights: test_unit.best_weights};
        }
        NetworkSerial { biases: self.biases, weights: self.weights }
    }
}