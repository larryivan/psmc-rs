pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

pub fn logsumexp(vals: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return max;
    }
    let mut sum = 0.0;
    for &v in vals {
        sum += (v - max).exp();
    }
    max + sum.ln()
}
