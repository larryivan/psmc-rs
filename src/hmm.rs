// HMM forward-backward and Viterbi will live here.
// This module is intentionally minimal for the first iteration.

use ndarray::Array2;

#[derive(Debug)]
pub struct ForwardBackwardResult {
    pub gamma: Array2<f64>,
    pub xi: Array2<f64>,
    pub c_norm: Vec<f64>,
}

// TODO: Implement forward-backward using scaled probabilities.
