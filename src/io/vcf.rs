use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use crate::io::Observations;

const N_RATIO: f64 = 0.9;

type Interval = (u64, u64); // inclusive [start, end], 1-based

#[derive(Debug, Clone, Default)]
struct BinAcc {
    len: usize,
    n_n: usize,
    has_k: bool,
}

fn flush_full_bin(seq: &mut Vec<u8>, bin: &mut BinAcc, bin_size: usize) {
    if bin.len != bin_size {
        return;
    }
    let sym = if (bin.n_n as f64) / (bin_size as f64) > N_RATIO {
        2
    } else if bin.has_k {
        1
    } else {
        0
    };
    seq.push(sym);
    *bin = BinAcc::default();
}

fn flush_tail_bin(seq: &mut Vec<u8>, bin: &mut BinAcc, bin_size: usize) {
    if bin.len == 0 {
        return;
    }
    let sym = if (bin.n_n as f64) / (bin_size as f64) > N_RATIO {
        2
    } else if bin.has_k {
        1
    } else {
        0
    };
    seq.push(sym);
    *bin = BinAcc::default();
}

fn add_run_t(seq: &mut Vec<u8>, mut n: u64, bin: &mut BinAcc, bin_size: usize) {
    while n > 0 {
        let room = (bin_size - bin.len) as u64;
        let take = n.min(room);
        bin.len += take as usize;
        n -= take;
        flush_full_bin(seq, bin, bin_size);
    }
}

fn add_run_n(seq: &mut Vec<u8>, mut n: u64, bin: &mut BinAcc, bin_size: usize) {
    while n > 0 {
        let room = (bin_size - bin.len) as u64;
        let take = n.min(room);
        bin.len += take as usize;
        bin.n_n += take as usize;
        n -= take;
        flush_full_bin(seq, bin, bin_size);
    }
}

fn add_k(seq: &mut Vec<u8>, bin: &mut BinAcc, bin_size: usize) {
    bin.len += 1;
    bin.has_k = true;
    flush_full_bin(seq, bin, bin_size);
}

fn chunk_sequences(seqs: Vec<Vec<u8>>, batch_size: Option<usize>) -> Result<Observations> {
    let mut rows = Vec::new();
    let mut row_starts = Vec::new();

    match batch_size {
        None => {
            for seq in seqs {
                if seq.is_empty() {
                    continue;
                }
                rows.push(seq);
                row_starts.push(true);
            }
        }
        Some(batch) => {
            if batch == 0 {
                bail!("batch_size must be > 0");
            }
            for seq in seqs {
                if seq.is_empty() {
                    continue;
                }
                for (i, chunk) in seq.chunks(batch).enumerate() {
                    rows.push(chunk.to_vec());
                    row_starts.push(i == 0);
                }
            }
        }
    }

    if rows.is_empty() {
        bail!("no valid segregating sites found in VCF input");
    }
    Ok(Observations { rows, row_starts })
}

fn open_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to open {path:?}"))?;
    let reader: Box<dyn Read> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::new(reader)))
}

fn merge_intervals(mut xs: Vec<Interval>) -> Vec<Interval> {
    if xs.is_empty() {
        return xs;
    }
    xs.sort_unstable_by_key(|x| (x.0, x.1));
    let mut out = Vec::with_capacity(xs.len());
    let mut cur = xs[0];
    for &(s, e) in &xs[1..] {
        if s <= cur.1.saturating_add(1) {
            cur.1 = max(cur.1, e);
        } else {
            out.push(cur);
            cur = (s, e);
        }
    }
    out.push(cur);
    out
}

fn intersect_intervals(a: &[Interval], b: &[Interval]) -> Vec<Interval> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut out = Vec::<Interval>::new();
    while i < a.len() && j < b.len() {
        let (as_, ae) = a[i];
        let (bs, be) = b[j];
        let s = max(as_, bs);
        let e = min(ae, be);
        if s <= e {
            out.push((s, e));
        }
        if ae < be {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

fn subtract_intervals(a: &[Interval], b: &[Interval]) -> Vec<Interval> {
    if a.is_empty() {
        return Vec::new();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let mut out = Vec::<Interval>::new();
    let mut j = 0usize;
    for &(as_, ae) in a {
        let mut cur_s = as_;
        while j < b.len() && b[j].1 < cur_s {
            j += 1;
        }
        let mut k = j;
        while k < b.len() && b[k].0 <= ae {
            let (bs, be) = b[k];
            if bs > cur_s {
                out.push((cur_s, bs - 1));
            }
            if be >= ae {
                cur_s = ae.saturating_add(1);
                break;
            }
            cur_s = be.saturating_add(1);
            k += 1;
        }
        if cur_s <= ae {
            out.push((cur_s, ae));
        }
    }
    out
}

fn overlap_len(intervals: &[Interval], start: u64, end: u64) -> u64 {
    if intervals.is_empty() || start > end {
        return 0;
    }
    let mut lo = 0usize;
    let mut hi = intervals.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if intervals[mid].1 < start {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    let mut total = 0u64;
    let mut i = lo;
    while i < intervals.len() {
        let (s, e) = intervals[i];
        if s > end {
            break;
        }
        let ov_s = max(s, start);
        let ov_e = min(e, end);
        if ov_s <= ov_e {
            total += ov_e - ov_s + 1;
        }
        i += 1;
    }
    total
}

#[derive(Debug, Clone, Default)]
struct MaskFile {
    by_chrom: HashMap<String, Vec<Interval>>,
    global: Vec<Interval>, // 2-col mask lines apply globally
}

impl MaskFile {
    fn is_empty(&self) -> bool {
        self.by_chrom.is_empty() && self.global.is_empty()
    }

    fn merge_all(&mut self) {
        for v in self.by_chrom.values_mut() {
            let merged = merge_intervals(std::mem::take(v));
            *v = merged;
        }
        self.global = merge_intervals(std::mem::take(&mut self.global));
    }

    fn union_from(paths: &[PathBuf]) -> Result<Self> {
        let mut out = MaskFile::default();
        for p in paths {
            let one = parse_mask_file(p)?;
            out.global.extend(one.global);
            for (chrom, intervals) in one.by_chrom {
                out.by_chrom.entry(chrom).or_default().extend(intervals);
            }
        }
        out.merge_all();
        Ok(out)
    }

    fn intervals_for_chrom(&self, chrom: &str) -> Vec<Interval> {
        let mut out = Vec::<Interval>::new();
        if let Some(v) = self.by_chrom.get(chrom) {
            out.extend(v.iter().copied());
        }
        if !self.global.is_empty() {
            out.extend(self.global.iter().copied());
        }
        merge_intervals(out)
    }
}

fn parse_mask_file(path: &Path) -> Result<MaskFile> {
    let mut reader = open_reader(path)?;
    let mut line = String::new();
    let mut out = MaskFile::default();
    let mut row_no = 0usize;

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .with_context(|| format!("failed to read mask {path:?}"))?;
        if n == 0 {
            break;
        }
        row_no += 1;
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = s.split_whitespace().collect();
        if cols.len() < 2 {
            bail!(
                "invalid mask row {} in {}: expected 2 or 3+ columns",
                row_no,
                path.display()
            );
        }

        if cols.len() == 2 {
            // Two-column mask format: start end (1-based inclusive).
            let start: u64 = cols[0].parse().with_context(|| {
                format!(
                    "invalid mask row {} in {}: bad start '{}'",
                    row_no,
                    path.display(),
                    cols[0]
                )
            })?;
            let end: u64 = cols[1].parse().with_context(|| {
                format!(
                    "invalid mask row {} in {}: bad end '{}'",
                    row_no,
                    path.display(),
                    cols[1]
                )
            })?;
            if start == 0 || end < start {
                bail!(
                    "invalid mask row {} in {}: start/end out of range",
                    row_no,
                    path.display()
                );
            }
            out.global.push((start, end));
        } else {
            // BED-like format: chrom start end where start is 0-based and end is 1-based exclusive.
            let chrom = cols[0].to_string();
            let start0: u64 = cols[1].parse().with_context(|| {
                format!(
                    "invalid mask row {} in {}: bad start '{}'",
                    row_no,
                    path.display(),
                    cols[1]
                )
            })?;
            let end_exclusive: u64 = cols[2].parse().with_context(|| {
                format!(
                    "invalid mask row {} in {}: bad end '{}'",
                    row_no,
                    path.display(),
                    cols[2]
                )
            })?;
            if end_exclusive <= start0 {
                bail!(
                    "invalid mask row {} in {}: end must be > start",
                    row_no,
                    path.display()
                );
            }
            let start = start0 + 1;
            let end = end_exclusive;
            out.by_chrom.entry(chrom).or_default().push((start, end));
        }
    }

    out.merge_all();
    Ok(out)
}

#[derive(Debug, Clone)]
enum MaskMode {
    AllCallable,
    AllExceptNegative,
    PositiveIntersection,
}

#[derive(Debug, Clone)]
struct MaskEngine {
    mode: MaskMode,
    positives: Vec<MaskFile>,
    negatives: MaskFile,
    pos_cache: HashMap<String, Vec<Interval>>, // final callable intervals for chrom
    neg_cache: HashMap<String, Vec<Interval>>, // negative intervals for chrom
}

impl MaskEngine {
    fn new(mask_paths: &[PathBuf], negative_mask_paths: &[PathBuf]) -> Result<Self> {
        let mut positives = Vec::<MaskFile>::new();
        for p in mask_paths {
            positives.push(parse_mask_file(p)?);
        }
        let negatives = MaskFile::union_from(negative_mask_paths)?;
        let mode = if positives.is_empty() {
            if negatives.is_empty() {
                MaskMode::AllCallable
            } else {
                MaskMode::AllExceptNegative
            }
        } else {
            MaskMode::PositiveIntersection
        };
        Ok(Self {
            mode,
            positives,
            negatives,
            pos_cache: HashMap::new(),
            neg_cache: HashMap::new(),
        })
    }

    fn negative_for_chrom(&mut self, chrom: &str) -> &[Interval] {
        if !self.neg_cache.contains_key(chrom) {
            let v = self.negatives.intervals_for_chrom(chrom);
            self.neg_cache.insert(chrom.to_string(), v);
        }
        self.neg_cache.get(chrom).map(Vec::as_slice).unwrap_or(&[])
    }

    fn callable_for_chrom(&mut self, chrom: &str) -> &[Interval] {
        if !self.pos_cache.contains_key(chrom) {
            let mut cur = self.positives[0].intervals_for_chrom(chrom);
            for m in &self.positives[1..] {
                let rhs = m.intervals_for_chrom(chrom);
                cur = intersect_intervals(&cur, &rhs);
                if cur.is_empty() {
                    break;
                }
            }
            let neg = self.negative_for_chrom(chrom);
            let final_intervals = subtract_intervals(&cur, neg);
            self.pos_cache.insert(chrom.to_string(), final_intervals);
        }
        self.pos_cache.get(chrom).map(Vec::as_slice).unwrap_or(&[])
    }

    fn count_callable(&mut self, chrom: &str, start: u64, end: u64) -> u64 {
        if start > end {
            return 0;
        }
        match self.mode {
            MaskMode::AllCallable => end - start + 1,
            MaskMode::AllExceptNegative => {
                let len = end - start + 1;
                let neg = self.negative_for_chrom(chrom);
                len.saturating_sub(overlap_len(neg, start, end))
            }
            MaskMode::PositiveIntersection => {
                let pos = self.callable_for_chrom(chrom);
                overlap_len(pos, start, end)
            }
        }
    }

    fn is_callable(&mut self, chrom: &str, pos: u64) -> bool {
        self.count_callable(chrom, pos, pos) > 0
    }
}

fn parse_gt(format_field: &str, sample_field: &str) -> Option<(usize, usize)> {
    let mut gt_index = None;
    for (i, key) in format_field.split(':').enumerate() {
        if key == "GT" {
            gt_index = Some(i);
            break;
        }
    }
    let idx = gt_index?;
    let gt = sample_field.split(':').nth(idx)?;
    if gt == "." || gt == "./." || gt == ".|." {
        return None;
    }
    let parts: Vec<&str> = gt.split(['/', '|']).collect();
    if parts.len() != 2 {
        return None;
    }
    if parts[0] == "." || parts[1] == "." {
        return None;
    }
    let a0 = parts[0].parse::<usize>().ok()?;
    let a1 = parts[1].parse::<usize>().ok()?;
    Some((a0, a1))
}

fn allele_from_index<'a>(ref_allele: &'a str, alt_field: &'a str, idx: usize) -> Option<&'a str> {
    if idx == 0 {
        return Some(ref_allele);
    }
    alt_field.split(',').nth(idx - 1)
}

fn resolve_sample_index(header_fields: &[&str], sample_name: Option<&str>) -> Result<usize> {
    if header_fields.len() < 10 {
        bail!("VCF must contain at least one sample column");
    }
    let samples = &header_fields[9..];
    if let Some(name) = sample_name {
        for (i, s) in samples.iter().enumerate() {
            if *s == name {
                return Ok(i);
            }
        }
        bail!("VCF sample '{}' not found in header", name);
    }
    if samples.len() == 1 {
        return Ok(0);
    }
    bail!(
        "VCF has multiple samples ({}) - please specify --vcf-sample",
        samples.len()
    )
}

pub fn read_vcf(
    path: &Path,
    batch_size: Option<usize>,
    bin_size: usize,
    sample_name: Option<&str>,
    mask_paths: &[PathBuf],
    negative_mask_paths: &[PathBuf],
) -> Result<Observations> {
    if bin_size == 0 {
        bail!("vcf bin_size must be > 0");
    }

    let mut reader = open_reader(path)?;
    let mut line = String::new();
    let mut row_no = 0usize;

    let mut sample_idx: Option<usize> = None;
    let mut masks = MaskEngine::new(mask_paths, negative_mask_paths)?;

    let mut seqs = Vec::<Vec<u8>>::new();
    let mut cur_seq = Vec::<u8>::new();
    let mut cur_bin = BinAcc::default();
    let mut cur_chrom: Option<String> = None;

    let mut prev_scan_pos: u64 = 0;
    let mut prev_emit_pos: u64 = 0;
    let mut called_since_emit: u64 = 0;
    let mut emitted_rows = 0usize;

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .with_context(|| format!("failed to read VCF {path:?}"))?;
        if n == 0 {
            break;
        }
        row_no += 1;
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if s.starts_with("##") {
            continue;
        }
        if s.starts_with("#CHROM") {
            let fields: Vec<&str> = s.split('\t').collect();
            sample_idx = Some(resolve_sample_index(&fields, sample_name)?);
            continue;
        }
        if s.starts_with('#') {
            continue;
        }
        let idx = sample_idx.with_context(|| "VCF header '#CHROM' not found before records")?;

        let fields: Vec<&str> = s.split('\t').collect();
        if fields.len() < 10 {
            bail!("invalid VCF row {}: expected >= 10 columns", row_no);
        }
        let sample_col = 9 + idx;
        if fields.len() <= sample_col {
            bail!(
                "invalid VCF row {}: missing sample column index {}",
                row_no,
                sample_col
            );
        }

        let chrom = fields[0];
        let pos: u64 = fields[1]
            .parse()
            .with_context(|| format!("invalid VCF row {}: bad POS '{}'", row_no, fields[1]))?;
        if pos == 0 {
            bail!("invalid VCF row {}: POS must be >= 1", row_no);
        }

        let is_new_chrom = match &cur_chrom {
            None => true,
            Some(c) => c != chrom,
        };
        if is_new_chrom {
            if cur_chrom.is_some() {
                flush_tail_bin(&mut cur_seq, &mut cur_bin, bin_size);
                if !cur_seq.is_empty() {
                    seqs.push(std::mem::take(&mut cur_seq));
                }
            }
            cur_chrom = Some(chrom.to_string());
            prev_scan_pos = 0;
            prev_emit_pos = 0;
            called_since_emit = 0;
            cur_bin = BinAcc::default();
        }

        if pos <= prev_scan_pos {
            bail!(
                "invalid VCF row {}: POS must increase within chromosome",
                row_no
            );
        }

        let called_between = masks.count_callable(chrom, prev_scan_pos + 1, pos);
        called_since_emit = called_since_emit.saturating_add(called_between);
        prev_scan_pos = pos;

        let format_field = fields[8];
        let sample_field = fields[sample_col];
        let Some((a0_idx, a1_idx)) = parse_gt(format_field, sample_field) else {
            continue;
        };
        let ref_allele = fields[3];
        let alt_field = fields[4];
        let Some(a0) = allele_from_index(ref_allele, alt_field, a0_idx) else {
            continue;
        };
        let Some(a1) = allele_from_index(ref_allele, alt_field, a1_idx) else {
            continue;
        };
        // Treat genotypes as heterozygous only if final allele strings differ.
        // This avoids false K sites when different ALT indices map to identical bases.
        if a0 == a1 {
            continue;
        }
        if !masks.is_callable(chrom, pos) {
            continue;
        }

        let delta = pos - prev_emit_pos;
        if called_since_emit == 0 {
            // Current segregating site should contribute at least one callable base.
            continue;
        }
        if called_since_emit > delta {
            bail!(
                "invalid derived mhs row at VCF row {}: nr_called ({}) exceeds distance ({})",
                row_no,
                called_since_emit,
                delta
            );
        }

        let n_uncalled = delta - called_since_emit;
        let n_called_non_variant = called_since_emit - 1;
        add_run_n(&mut cur_seq, n_uncalled, &mut cur_bin, bin_size);
        add_run_t(&mut cur_seq, n_called_non_variant, &mut cur_bin, bin_size);
        add_k(&mut cur_seq, &mut cur_bin, bin_size);

        prev_emit_pos = pos;
        called_since_emit = 0;
        emitted_rows += 1;
    }

    flush_tail_bin(&mut cur_seq, &mut cur_bin, bin_size);
    if !cur_seq.is_empty() {
        seqs.push(cur_seq);
    }

    if emitted_rows == 0 {
        bail!("no segregating callable diploid sites found in VCF input");
    }
    chunk_sequences(seqs, batch_size)
}
