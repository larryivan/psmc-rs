use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsmcParamsFile {
    pub theta: f64,
    pub rho: f64,
    pub lam: Vec<f64>,
    pub t_max: f64,
    pub n_steps: usize,
    pub mu: f64,
    pub pattern: Option<String>,
}

pub fn save_params(path: &Path, params: &PsmcParamsFile) -> Result<()> {
    let file = File::create(path).with_context(|| format!("failed to create {:?}", path))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, params)
        .with_context(|| format!("failed to write {:?}", path))?;
    Ok(())
}

pub fn load_params(path: &Path) -> Result<PsmcParamsFile> {
    let file = File::open(path).with_context(|| format!("failed to open {:?}", path))?;
    let reader = BufReader::new(file);
    let params =
        serde_json::from_reader(reader).with_context(|| format!("failed to parse {:?}", path))?;
    Ok(params)
}
