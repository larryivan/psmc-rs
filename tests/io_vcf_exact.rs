//! The VCF reader against an independent per-base reference: build the state
//! of every position straight from the definition, bin it with a literal port
//! of `fq2psmcfa`, and compare bin for bin over many random scenarios.

use psmc_rs::io::vcf::{VcfOptions, read_vcf, read_vcf_with};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

/// A path no other test of this process uses: tests run in parallel, and
/// the clock alone is too coarse to tell them apart.
fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let n = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()))
}

/// SplitMix64: deterministic and dependency-free.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `lo..=hi`.
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }

    fn chance(&mut self, p: f64) -> bool {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64) < p
    }

    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.range(0, xs.len() as u64 - 1) as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum State {
    Hom,
    Het,
    Missing,
}

#[derive(Clone, Debug)]
struct Record {
    chrom: usize,
    pos: u64,
    reference: String,
    /// Empty means ALT `.`.
    alts: Vec<String>,
    filter: String,
    /// `None`: FORMAT has no GT key; otherwise whether GT comes first.
    gt_first: Option<bool>,
    /// One genotype per sample.
    gts: Vec<String>,
}

#[derive(Clone, Debug)]
enum MaskRow {
    /// BED: 0-based half-open, one chromosome.
    Bed {
        chrom: String,
        start0: u64,
        end: u64,
    },
    /// Two columns: 1-based inclusive, every chromosome.
    Global { start: u64, end: u64 },
}

#[derive(Clone, Debug)]
struct Scenario {
    chroms: Vec<String>,
    /// `##contig` length of each chromosome, if the header gives one.
    contig_len: Vec<Option<u64>>,
    n_samples: usize,
    sample: usize,
    name_sample: bool,
    records: Vec<Record>,
    masks: Vec<Vec<MaskRow>>,
    negative_masks: Vec<Vec<MaskRow>>,
    bin_size: usize,
    batch_size: Option<usize>,
    keep_filtered: bool,
    to_contig_end: bool,
}

impl Scenario {
    /// The last position of chromosome `ci`'s sequence, if it has records.
    fn extent(&self, ci: usize) -> Option<u64> {
        let last = self
            .records
            .iter()
            .filter(|r| r.chrom == ci)
            .map(|r| r.pos)
            .max()?;
        match self.contig_len[ci] {
            Some(len) if self.to_contig_end => Some(len),
            _ => Some(last),
        }
    }
}

// ---------------------------------------------------------------- reference

/// Literal port of the binning loop of lh3/psmc `utils/fq2psmcfa.c`, run on a
/// sequence where `Missing` stands for N and `Het` for an IUPAC het code.
fn fq2psmcfa(states: &[State], block_len: usize) -> Vec<u8> {
    const N_RATIO: f64 = 0.9;
    let symbol = |n_n: usize, is_hetb: bool| {
        // `(float)nN/BLOCK_LEN > N_RATIO` in C: float division, double compare.
        if f64::from(n_n as f32 / block_len as f32) > N_RATIO {
            2
        } else if is_hetb {
            1
        } else {
            0
        }
    };
    let mut out = Vec::new();
    let (mut n_n, mut is_hetb) = (0usize, false);
    for (i, &s) in states.iter().enumerate() {
        if i != 0 && i % block_len == 0 {
            out.push(symbol(n_n, is_hetb));
            is_hetb = false;
            n_n = 0;
        }
        match s {
            State::Missing => n_n += 1,
            State::Het => is_hetb = true,
            State::Hom => {}
        }
    }
    out.push(symbol(n_n, is_hetb));
    out
}

/// `covered[p]` for `p` in `0..=len`: whether a row of `rows` covers position
/// `p` (1-based) of `chrom`.
fn paint(rows: &[MaskRow], chrom: &str, len: u64) -> Vec<bool> {
    let mut covered = vec![false; len as usize + 1];
    for row in rows {
        let (start, end) = match row {
            MaskRow::Bed {
                chrom: c,
                start0,
                end,
            } if c == chrom => (start0 + 1, *end),
            MaskRow::Bed { .. } => continue,
            MaskRow::Global { start, end } => (*start, *end),
        };
        for p in start..=end.min(len) {
            covered[p as usize] = true;
        }
    }
    covered
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    MissingGt,
    Hom,
    HetSnp,
    HetOther,
}

fn is_filtered(rec: &Record) -> bool {
    rec.filter != "PASS" && rec.filter != "."
}

/// What the selected sample's genotype says, ignoring FILTER.
fn genotype_kind(rec: &Record, sample: usize) -> Kind {
    if rec.gt_first.is_none() {
        return Kind::MissingGt;
    }
    let indices: Vec<&str> = rec.gts[sample].split(['/', '|']).collect();
    if indices.contains(&".") {
        return Kind::MissingGt;
    }
    let alleles: Vec<String> = indices
        .iter()
        .map(|i| {
            let i: usize = i.parse().expect("test genotype index");
            let a = if i == 0 {
                &rec.reference
            } else {
                &rec.alts[i - 1]
            };
            a.to_ascii_uppercase()
        })
        .collect();
    if alleles.iter().all(|a| *a == alleles[0]) {
        Kind::Hom
    } else if alleles
        .iter()
        .all(|a| ["A", "C", "G", "T"].contains(&a.as_str()))
    {
        Kind::HetSnp
    } else {
        Kind::HetOther
    }
}

/// The class a record gives its POS, and whether its whole REF span is missing.
fn record_class(rec: &Record, sample: usize, keep_filtered: bool) -> (State, bool) {
    if is_filtered(rec) && !keep_filtered {
        return (State::Missing, true);
    }
    match genotype_kind(rec, sample) {
        Kind::MissingGt | Kind::HetOther => (State::Missing, true),
        Kind::Hom => (State::Hom, false),
        Kind::HetSnp => (State::Het, false),
    }
}

/// Per-base states of every chromosome with records, index 0 unused.
fn reference_states(sc: &Scenario) -> Vec<Vec<State>> {
    let mut out = Vec::new();
    for (ci, name) in sc.chroms.iter().enumerate() {
        let recs: Vec<&Record> = sc.records.iter().filter(|r| r.chrom == ci).collect();
        let Some(len) = sc.extent(ci) else {
            continue;
        };
        let n = len as usize;
        let positives: Vec<Vec<bool>> = sc.masks.iter().map(|m| paint(m, name, len)).collect();
        let negatives: Vec<Vec<bool>> = sc
            .negative_masks
            .iter()
            .map(|m| paint(m, name, len))
            .collect();
        let mut st = vec![State::Hom; n + 1];
        for (p, s) in st.iter_mut().enumerate().skip(1) {
            let callable = positives.iter().all(|m| m[p]) && !negatives.iter().any(|m| m[p]);
            if !callable {
                *s = State::Missing;
            }
        }
        for r in &recs {
            let (class, span) = record_class(r, sc.sample, sc.keep_filtered);
            let p = r.pos as usize;
            st[p] = st[p].max(class);
            if span {
                for s in &mut st[p..(p + r.reference.len()).min(n + 1)] {
                    *s = State::Missing;
                }
            }
        }
        out.push(st);
    }
    out
}

// ------------------------------------------------------------------ writers

fn vcf_text(sc: &Scenario) -> String {
    let mut s = String::from("##fileformat=VCFv4.2\n");
    for (name, len) in sc.chroms.iter().zip(&sc.contig_len) {
        match len {
            Some(len) => writeln!(s, "##contig=<ID={name},length={len}>").unwrap(),
            None => writeln!(s, "##contig=<ID={name}>").unwrap(),
        }
    }
    // A contig without records yields no sequence.
    s.push_str("##contig=<ID=chrEmpty,length=5000,assembly=\"test, v1\">\n");
    s.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT");
    for i in 0..sc.n_samples {
        write!(s, "\ts{i}").unwrap();
    }
    s.push('\n');
    for r in &sc.records {
        let alt = if r.alts.is_empty() {
            ".".to_string()
        } else {
            r.alts.join(",")
        };
        let format = match r.gt_first {
            None => "DP",
            Some(true) => "GT:DP",
            Some(false) => "DP:GT",
        };
        write!(
            s,
            "{}\t{}\t.\t{}\t{}\t50\t{}\t.\t{}",
            sc.chroms[r.chrom], r.pos, r.reference, alt, r.filter, format
        )
        .unwrap();
        for g in &r.gts {
            match r.gt_first {
                None => s.push_str("\t7"),
                Some(true) => write!(s, "\t{g}:7").unwrap(),
                Some(false) => write!(s, "\t7:{g}").unwrap(),
            }
        }
        s.push('\n');
    }
    s
}

fn mask_text(rows: &[MaskRow]) -> String {
    let mut s = String::from("# test mask\n");
    for (i, row) in rows.iter().enumerate() {
        match row {
            MaskRow::Bed { chrom, start0, end } if i % 3 == 0 => {
                writeln!(s, "{chrom}\t{start0}\t{end}\tname{i}").unwrap()
            }
            MaskRow::Bed { chrom, start0, end } => writeln!(s, "{chrom}\t{start0}\t{end}").unwrap(),
            MaskRow::Global { start, end } => writeln!(s, "{start}\t{end}").unwrap(),
        }
    }
    s
}

struct Files {
    paths: Vec<PathBuf>,
}

impl Drop for Files {
    fn drop(&mut self) {
        for p in &self.paths {
            let _ = fs::remove_file(p);
        }
    }
}

/// Write the scenario's VCF and masks; returns the VCF path, the options to
/// read it with, and a guard that deletes the files.
fn write_scenario(sc: &Scenario) -> (PathBuf, VcfOptions, Files) {
    let mut files = Files { paths: Vec::new() };
    let mut write = |prefix: &str, ext: &str, text: String| {
        let p = unique_temp_path(prefix, ext);
        fs::write(&p, text).expect("failed to write test input");
        files.paths.push(p.clone());
        p
    };
    let vcf = write("exact_vcf", "vcf", vcf_text(sc));
    let masks = sc
        .masks
        .iter()
        .map(|m| write("exact_mask", "bed", mask_text(m)))
        .collect();
    let negative_masks = sc
        .negative_masks
        .iter()
        .map(|m| write("exact_neg", "bed", mask_text(m)))
        .collect();
    let opts = VcfOptions {
        bin_size: sc.bin_size,
        batch_size: sc.batch_size,
        sample: sc.name_sample.then(|| format!("s{}", sc.sample)),
        masks,
        negative_masks,
        keep_filtered: sc.keep_filtered,
        to_contig_end: sc.to_contig_end,
    };
    (vcf, opts, files)
}

fn chunk(seqs: &[Vec<u8>], batch_size: Option<usize>) -> (Vec<Vec<u8>>, Vec<bool>) {
    let mut rows = Vec::new();
    let mut starts = Vec::new();
    for s in seqs {
        let b = batch_size.unwrap_or(s.len());
        for (i, c) in s.chunks(b).enumerate() {
            rows.push(c.to_vec());
            starts.push(i == 0);
        }
    }
    (rows, starts)
}

/// Panic with the first difference and the share of differing bins.
fn assert_same_bins(got: &[Vec<u8>], want: &[Vec<u8>], ctx: &str) {
    if got == want {
        return;
    }
    let g: Vec<u8> = got.concat();
    let w: Vec<u8> = want.concat();
    let differ = g.iter().zip(&w).filter(|(a, b)| a != b).count() + g.len().abs_diff(w.len());
    let first = g
        .iter()
        .zip(&w)
        .position(|(a, b)| a != b)
        .unwrap_or(g.len().min(w.len()));
    panic!(
        "{ctx}: {differ} of {} bins differ ({:.1}%), first at bin {first}; rows {} vs {} \
         (lengths {:?} vs {:?})",
        w.len(),
        100.0 * differ as f64 / w.len().max(1) as f64,
        got.len(),
        want.len(),
        got.iter().map(Vec::len).collect::<Vec<_>>(),
        want.iter().map(Vec::len).collect::<Vec<_>>(),
    );
}

/// Read the scenario with the reader and compare everything with the
/// reference. Returns whether the reader produced output (the reference has a
/// heterozygous site).
fn check(sc: &Scenario, ctx: &str) -> bool {
    let states = reference_states(sc);
    let seqs: Vec<Vec<u8>> = states
        .iter()
        .map(|st| fq2psmcfa(&st[1..], sc.bin_size))
        .collect();
    let count = |want: State| -> u64 {
        states
            .iter()
            .map(|st| st[1..].iter().filter(|&&s| s == want).count() as u64)
            .sum()
    };
    let (vcf, opts, _files) = write_scenario(sc);
    let result = read_vcf_with(&vcf, &opts);
    if count(State::Het) == 0 {
        let err = result.expect_err(ctx);
        assert!(
            format!("{err:#}").contains("no heterozygous SNP"),
            "{ctx}: {err:#}"
        );
        return false;
    }
    let read = result.unwrap_or_else(|e| panic!("{ctx}: {e:#}"));
    let (rows, starts) = chunk(&seqs, sc.batch_size);
    assert_same_bins(&read.obs.rows, &rows, ctx);
    assert_eq!(read.obs.row_starts, starts, "{ctx}");

    let s = &read.summary;
    assert_eq!(s.het_snps, count(State::Het), "{ctx}: het_snps");
    assert_eq!(
        s.missing_sites,
        count(State::Missing),
        "{ctx}: missing_sites"
    );
    let sites: u64 = states.iter().map(|st| st.len() as u64 - 1).sum();
    assert_eq!(s.sites, sites, "{ctx}: sites");
    assert_eq!(s.sequences, seqs.len(), "{ctx}: sequences");
    assert_eq!(s.bins, seqs.iter().map(Vec::len).sum::<usize>(), "{ctx}");
    assert_eq!(s.records, sc.records.len() as u64, "{ctx}: records");
    assert_eq!(s.positive_mask, !sc.masks.is_empty(), "{ctx}");
    let extended: Vec<&String> = sc
        .chroms
        .iter()
        .zip(&sc.contig_len)
        .filter(|(_, len)| sc.to_contig_end && len.is_some())
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        s.contig_lengths_used.iter().collect::<Vec<_>>(),
        extended,
        "{ctx}"
    );

    let n_filtered = sc.records.iter().filter(|r| is_filtered(r)).count() as u64;
    assert_eq!(s.filtered_records, n_filtered, "{ctx}: filtered_records");
    let n_kind = |kind: Kind| -> u64 {
        sc.records
            .iter()
            .filter(|r| {
                (sc.keep_filtered || !is_filtered(r)) && genotype_kind(r, sc.sample) == kind
            })
            .count() as u64
    };
    assert_eq!(s.missing_genotypes, n_kind(Kind::MissingGt), "{ctx}");
    assert_eq!(s.hom_calls, n_kind(Kind::Hom), "{ctx}: hom_calls");
    assert_eq!(s.het_snp_records, n_kind(Kind::HetSnp), "{ctx}");
    assert_eq!(s.het_non_snp_records, n_kind(Kind::HetOther), "{ctx}");
    true
}

// ---------------------------------------------------------------- scenarios

const BASES: [&str; 4] = ["A", "C", "G", "T"];

fn other_base(rng: &mut Rng, not: &[&str]) -> &'static str {
    loop {
        let b = rng.pick(&BASES);
        if !not.contains(&b) {
            return b;
        }
    }
}

/// A random record at `pos` whose genotype for `sample` covers every case of
/// the classification rules.
fn random_record(rng: &mut Rng, chrom: usize, pos: u64, n_samples: usize, sample: usize) -> Record {
    let base = rng.pick(&BASES);
    let b2 = other_base(rng, &[base]);
    let b3 = other_base(rng, &[base, b2]);
    let s = |x: &str| x.to_string();
    let het = ["0/1", "1/0", "0|1", "1|0"];
    let mut gt_first = Some(rng.chance(0.8));
    let (reference, alts, gt): (String, Vec<String>, String) = match rng.range(0, 99) {
        // Heterozygous SNPs.
        0..=29 => (s(base), vec![s(b2)], s(rng.pick(&het))),
        30..=33 => (s(base), vec![s(b2), s(b3)], s("1/2")),
        34..=37 => (base.to_lowercase(), vec![b2.to_lowercase()], s("0/1")),
        // Homozygous and haploid calls.
        38..=47 => (s(base), vec![s(b2)], s(rng.pick(&["0/0", "0|0"]))),
        48..=51 => (s(base), vec![], s("0/0")),
        52..=58 => (s(base), vec![s(b2)], s("1/1")),
        59..=61 => (s(base), vec![s(b2)], s(rng.pick(&["0", "1"]))),
        62..=64 => (s(base), vec![s(b2), s(b2)], s("1/2")),
        65..=67 => (format!("{base}{b2}{b3}"), vec![s(base)], s("1/1")),
        // Missing or partial genotypes.
        68..=75 => (
            s(base),
            vec![s(b2)],
            s(rng.pick(&["./.", ".", ".|.", "./1", "0/."])),
        ),
        76..=79 => (format!("{base}{b2}{b3}{b2}"), vec![s(base)], s("./.")),
        // Heterozygous indels, spanning deletions and symbolic alleles.
        80..=85 => {
            let extra = rng.range(1, 3) as usize;
            let long: String = std::iter::repeat_n(b2, extra).collect();
            if rng.chance(0.5) {
                (format!("{base}{long}"), vec![s(base)], s("0/1"))
            } else {
                (s(base), vec![format!("{base}{long}")], s("1|0"))
            }
        }
        86..=89 => (s(base), vec![s(b2), s("*")], s("1/2")),
        90..=92 => (s(base), vec![s("<DEL>")], s("0/1")),
        93..=95 => (format!("{base}{b2}"), vec![s(b3), s(base)], s("1/2")),
        // No GT in FORMAT.
        _ => {
            gt_first = None;
            (format!("{base}{b2}"), vec![s(base)], s("0/1"))
        }
    };
    let filter = if rng.chance(0.15) {
        s(rng.pick(&["LowQual", "q10;s50"]))
    } else {
        s(rng.pick(&["PASS", "."]))
    };
    let gts = (0..n_samples)
        .map(|i| {
            if i == sample {
                gt.clone()
            } else {
                s(rng.pick(&["0/0", "./.", "0|0"]))
            }
        })
        .collect();
    Record {
        chrom,
        pos,
        reference,
        alts,
        filter,
        gt_first,
        gts,
    }
}

/// Alternating in/out runs over `1..=len` with the given run-length ranges;
/// returns the "in" runs as 1-based inclusive intervals.
fn runs(rng: &mut Rng, len: u64, in_len: (u64, u64), out_len: (u64, u64)) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut p = 1;
    let mut inside = rng.chance(0.5);
    while p <= len {
        let (lo, hi) = if inside { in_len } else { out_len };
        let r = rng.range(lo, hi);
        if inside {
            out.push((p, p + r - 1));
        }
        p += r;
        inside = !inside;
    }
    out
}

/// Random mask rows for chromosomes of the given lengths, mixing dispersed
/// and clustered runs, skipped chromosomes, rows for absent chromosomes,
/// overlapping rows and two-column rows.
fn random_mask(rng: &mut Rng, chroms: &[String], lens: &[u64], negative: bool) -> Vec<MaskRow> {
    let mut rows = Vec::new();
    for (name, &len) in chroms.iter().zip(lens) {
        if rng.chance(0.1) {
            continue;
        }
        let ivs = match rng.range(0, 3) {
            0 if negative => runs(rng, len, (1, 3), (1, 12)),
            0 => runs(rng, len, (1, 12), (1, 3)),
            1 if negative => runs(rng, len, (5, 80), (50, 900)),
            1 => runs(rng, len, (50, 900), (5, 80)),
            2 => runs(rng, len, (1, 40), (1, 40)),
            _ if negative => vec![],
            _ => vec![(1, len + rng.range(0, 50))],
        };
        for (s, e) in ivs {
            rows.push(MaskRow::Bed {
                chrom: name.clone(),
                start0: s - 1,
                end: e,
            });
            if rng.chance(0.05) {
                let s2 = rng.range(s, e);
                rows.push(MaskRow::Bed {
                    chrom: name.clone(),
                    start0: s2 - 1,
                    end: e + rng.range(0, 5),
                });
            }
        }
    }
    if rng.chance(0.2) {
        rows.push(MaskRow::Bed {
            chrom: "chrUn".to_string(),
            start0: 0,
            end: 1000,
        });
    }
    if rng.chance(0.15) {
        for _ in 0..rng.range(1, 3) {
            let start = rng.range(1, 2000);
            rows.push(MaskRow::Global {
                start,
                end: start + rng.range(0, 300),
            });
        }
    }
    // Rows need not be sorted.
    if rng.chance(0.3) {
        for i in (1..rows.len()).rev() {
            let j = rng.range(0, i as u64) as usize;
            rows.swap(i, j);
        }
    }
    rows
}

fn random_scenario(seed: u64) -> Scenario {
    let mut rng = Rng(seed);
    let n_chroms = rng.range(1, 3) as usize;
    let chroms: Vec<String> = (0..n_chroms).map(|i| format!("chr{}", i + 1)).collect();
    let n_samples = rng.range(1, 3) as usize;
    let sample = rng.range(0, n_samples as u64 - 1) as usize;
    let mut records = Vec::new();
    let mut lens = Vec::new();
    let mut contig_len = Vec::new();
    for ci in 0..n_chroms {
        let span = rng.range(20, 2500);
        let density = rng.pick(&[0.002, 0.02, 0.2, 0.9]);
        let mut positions = Vec::new();
        for p in 1..=span {
            if rng.chance(density) {
                positions.push(p);
                while rng.chance(0.1) {
                    positions.push(p);
                }
            }
        }
        if positions.is_empty() {
            positions.push(rng.range(1, span));
        }
        let last = *positions.last().unwrap();
        let tail = rng.range(2, 400);
        let tail = rng.pick(&[0, 1, tail]);
        let len = rng.chance(0.5).then_some(last + tail);
        contig_len.push(len);
        lens.push(len.unwrap_or(last));
        for p in positions {
            records.push(random_record(&mut rng, ci, p, n_samples, sample));
        }
    }
    let n_masks = rng.pick(&[0, 0, 1, 1, 2]);
    let masks = (0..n_masks)
        .map(|_| random_mask(&mut rng, &chroms, &lens, false))
        .collect();
    let n_neg = rng.pick(&[0, 1, 1, 2]);
    let negative_masks = (0..n_neg)
        .map(|_| random_mask(&mut rng, &chroms, &lens, true))
        .collect();
    Scenario {
        chroms,
        contig_len,
        n_samples,
        sample,
        name_sample: n_samples > 1 || rng.chance(0.5),
        records,
        masks,
        negative_masks,
        bin_size: rng.pick(&[1, 7, 100]),
        batch_size: rng.pick(&[None, None, Some(1), Some(3), Some(40)]),
        keep_filtered: rng.chance(0.3),
        to_contig_end: rng.chance(0.8),
    }
}

#[test]
fn reader_matches_per_base_reference_on_random_scenarios() {
    let n = 400;
    let mut with_output = 0;
    for seed in 0..n {
        let sc = random_scenario(seed);
        if check(&sc, &format!("scenario seed {seed}")) {
            with_output += 1;
        }
    }
    // Most scenarios must exercise the comparison, not just the error path.
    assert!(with_output > n * 3 / 4, "{with_output} of {n}");
}

// ------------------------------------------------------------ audit scenarios

/// The audit's 1 Mb benchmark: het SNPs at 1/1000 and hom-alt calls at
/// 1/2000, with the given per-base missingness supplied as a callable mask.
fn audit_scenario(seed: u64, missing: impl Fn(&mut Rng, u64) -> Vec<bool>) -> Scenario {
    const LEN: u64 = 1_000_000;
    let mut rng = Rng(seed);
    let miss = missing(&mut rng, LEN);
    let mut mask = Vec::new();
    let mut p = 1;
    while p <= LEN {
        if miss[p as usize] {
            p += 1;
            continue;
        }
        let start = p;
        while p <= LEN && !miss[p as usize] {
            p += 1;
        }
        mask.push(MaskRow::Bed {
            chrom: "chr1".to_string(),
            start0: start - 1,
            end: p - 1,
        });
    }
    let record = |pos: u64, alt: &str, gt: &str| Record {
        chrom: 0,
        pos,
        reference: "A".to_string(),
        alts: vec![alt.to_string()],
        filter: "PASS".to_string(),
        gt_first: Some(true),
        gts: vec![gt.to_string()],
    };
    let mut records = Vec::new();
    for pos in 1..LEN {
        if rng.chance(0.001) {
            records.push(record(pos, "C", "0/1"));
        } else if rng.chance(0.0005) {
            records.push(record(pos, "G", "1/1"));
        }
    }
    records.push(record(LEN, "C", "0/1"));
    Scenario {
        chroms: vec!["chr1".to_string()],
        contig_len: vec![None],
        n_samples: 1,
        sample: 0,
        name_sample: false,
        records,
        masks: vec![mask],
        negative_masks: vec![],
        bin_size: 100,
        batch_size: None,
        keep_filtered: false,
        to_contig_end: true,
    }
}

/// Compare `read_vcf` (the wrapper the CLI uses) with the reference.
fn check_audit(sc: &Scenario, ctx: &str) {
    let want: Vec<Vec<u8>> = reference_states(sc)
        .iter()
        .map(|st| fq2psmcfa(&st[1..], sc.bin_size))
        .collect();
    let (vcf, opts, _files) = write_scenario(sc);
    let got = read_vcf(&vcf, None, sc.bin_size, None, &opts.masks, &[])
        .unwrap_or_else(|e| panic!("{ctx}: {e:#}"));
    assert_same_bins(&got.rows, &want, ctx);
}

#[test]
fn audit_20pct_random_missing() {
    let sc = audit_scenario(1, |rng, len| {
        (0..=len).map(|p| p == 0 || rng.chance(0.2)).collect()
    });
    check_audit(&sc, "20% random missing");
}

#[test]
fn audit_50pct_missing_in_10bp_runs() {
    let sc = audit_scenario(2, |rng, len| {
        let blocks: Vec<bool> = (0..=len / 10).map(|_| rng.chance(0.5)).collect();
        (0..=len)
            .map(|p| p == 0 || blocks[(p / 10) as usize])
            .collect()
    });
    check_audit(&sc, "50% missing in 10 bp runs");
}

#[test]
fn audit_all_callable() {
    let sc = audit_scenario(3, |_, len| (0..=len).map(|p| p == 0).collect());
    check_audit(&sc, "all callable");
}

#[test]
fn audit_50pct_missing_in_50kb_blocks() {
    let sc = audit_scenario(4, |rng, len| {
        let blocks: Vec<bool> = (0..=len / 50_000).map(|_| rng.chance(0.5)).collect();
        (0..=len)
            .map(|p| p == 0 || blocks[(p / 50_000) as usize])
            .collect()
    });
    check_audit(&sc, "50% missing in 50 kb blocks");
}
