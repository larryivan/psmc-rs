pub mod mhs;
pub mod params;
pub mod psmcfa;
pub mod vcf;

#[derive(Debug, Clone)]
pub struct Observations {
    pub rows: Vec<Vec<u8>>,
    pub row_starts: Vec<bool>,
}
