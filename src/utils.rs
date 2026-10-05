pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}
