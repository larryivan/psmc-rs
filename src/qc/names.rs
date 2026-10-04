//! Recognising sex chromosomes and organelle genomes by sequence name.

use serde::Serialize;

/// What a sequence name says about the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SeqKind {
    /// A sex chromosome: X, Y, Z or W, possibly numbered (X1, Y2, ...).
    Sex,
    /// A mitochondrial or plastid genome.
    Organelle,
}

impl SeqKind {
    /// `"sex chromosome"` or `"organelle"`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Sex => "sex chromosome",
            Self::Organelle => "organelle",
        }
    }
}

/// Words that name an organelle wherever they appear, e.g. in
/// `dmel_mitochondrion_genome` or `NC_037304_chloroplast`.
const ORGANELLE_WORDS: &[&str] = &[
    "mitochondri",
    "chloroplast",
    "plastid",
    "plastome",
    "apicoplast",
];

/// RefSeq accessions (without version) of the human and mouse sex
/// chromosomes and mitochondria, which NCBI-based pipelines use as names.
const REFSEQ: &[(&str, SeqKind)] = &[
    ("nc_000023", SeqKind::Sex),       // human X
    ("nc_000024", SeqKind::Sex),       // human Y
    ("nc_012920", SeqKind::Organelle), // human MT (rCRS)
    ("nc_001807", SeqKind::Organelle), // human MT (hg19 chrM)
    ("nc_000086", SeqKind::Sex),       // mouse X
    ("nc_000087", SeqKind::Sex),       // mouse Y
    ("nc_005089", SeqKind::Organelle), // mouse MT
];

/// Classify a sequence name, ignoring case.
///
/// After an optional `chr`, `chrom` or `chromosome` prefix (and one `_`,
/// `-` or `.` after it), the leading run of letters and digits is compared:
///
/// - `X`, `Y`, `Z`, `W`, or one of them followed by digits (`X1`, `Y2`, as in
///   multiple sex chromosome systems): [`SeqKind::Sex`]. `chrY_KI270740v1_random`
///   is therefore a sex chromosome too.
/// - `M`, `MT`, `Mt`, `mito`, `mtDNA`, `Pt` (Ensembl Plants' chloroplast),
///   `cp`, and `C` after a `chr` prefix (TAIR's `ChrC`): [`SeqKind::Organelle`].
///
/// Names containing `mitochondri`, `chloroplast`, `plastid`, `plastome` or
/// `apicoplast` anywhere are organelles, and so are a few RefSeq accessions
/// (human and mouse X, Y and MT, e.g. `NC_012920.1`). Everything else, e.g.
/// `chr1`, `chrX1A`, `scaffold_12` or `chrUn_KI270302v1`, is `None`.
pub fn classify_name(name: &str) -> Option<SeqKind> {
    let lower = name.to_ascii_lowercase();
    if ORGANELLE_WORDS.iter().any(|w| lower.contains(w)) {
        return Some(SeqKind::Organelle);
    }
    let accession = lower.split('.').next().unwrap_or("");
    if let Some(&(_, kind)) = REFSEQ.iter().find(|(acc, _)| *acc == accession) {
        return Some(kind);
    }

    let (rest, prefixed) = ["chromosome", "chrom", "chr"]
        .iter()
        .find_map(|p| lower.strip_prefix(p))
        .map_or((lower.as_str(), false), |r| (r, true));
    let rest = if prefixed {
        rest.strip_prefix(['_', '-', '.']).unwrap_or(rest)
    } else {
        rest
    };
    let end = rest
        .find(|c: char| !c.is_ascii_alphanumeric())
        .unwrap_or(rest.len());
    let token = &rest[..end];

    let mut chars = token.chars();
    let first = chars.next()?;
    let tail = chars.as_str();
    if matches!(first, 'x' | 'y' | 'z' | 'w') && tail.bytes().all(|b| b.is_ascii_digit()) {
        return Some(SeqKind::Sex);
    }
    match token {
        "m" | "mt" | "mito" | "mtdna" | "pt" | "cp" => Some(SeqKind::Organelle),
        "c" if prefixed => Some(SeqKind::Organelle),
        _ => None,
    }
}
