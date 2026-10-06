//! The callable bases each reader gives every bin: the binner's count, the
//! multihetsep placement of uncalled sites, the loader (VCF and multihetsep
//! input with counts, psmcfa without), bins too large to count, the memory
//! the counts take, and the writers, which stay as they were. The VCF
//! reader's counts are checked bin for bin against a per-base reference in
//! `tests/io_vcf_exact.rs`.

use psmc_rs::io::common::Binner;
use psmc_rs::io::input::{Dataset, InputFormat, InputOptions, SeqFilter, VcfMode, load};
use psmc_rs::io::mhs::{MhsOptions, read_mhs_with};
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::io::vcf::{VcfOptions, read_vcf_with};
use psmc_rs::io::write::write_psmcfa;
use psmc_rs::io::{MAX_COUNTED_BIN_SIZE, Observations};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A temporary file removed on drop; tests run in parallel, so each gets a
/// name of its own.
struct TempFile(PathBuf);

impl TempFile {
    fn new(prefix: &str, ext: &str, content: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()));
        fs::write(&path, content).expect("failed to write temp file");
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
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
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Site {
    Called,
    Het,
    Missing,
}

/// `fq2psmcfa`'s bins of `sites` (N when more than 90% of `s` are missing,
/// a last block judged against the full `s`) and each bin's callable bases,
/// `s` minus its missing sites.
fn reference_bins(sites: &[Site], s: usize) -> (Vec<u8>, Vec<u16>) {
    sites
        .chunks(s)
        .map(|block| {
            let missing = block.iter().filter(|&&x| x == Site::Missing).count();
            let symbol = if 10 * missing > 9 * s {
                2
            } else {
                u8::from(block.contains(&Site::Het))
            };
            (symbol, u16::try_from(s - missing).unwrap())
        })
        .unzip()
}

/// The bins of the sequence `b` binned and their counts, one per bin.
fn finish(b: &mut Binner) -> (Vec<u8>, Option<Vec<u16>>) {
    let (bins, counts) = b.finish_with_callable();
    (bins, counts.map(|c| c.to_vec()))
}

#[test]
fn the_binner_counts_bin_size_minus_missing_sites() {
    let mut b = Binner::new(10);
    b.add_called(25);
    b.add_missing(3);
    b.add_het();
    b.add_missing(8);
    // Sites 21-30 hold 4 missing ones; the last bin, sites 31-37, is all
    // missing, but judged against 10 sites it is T, and its 3 sites past the
    // end count as callable.
    assert_eq!(finish(&mut b), (vec![0, 0, 1, 0], Some(vec![10, 10, 6, 3])));
    assert!(b.is_empty());
    b.add_missing(10);
    b.add_called(1);
    assert_eq!(finish(&mut b), (vec![2, 0], Some(vec![0, 10])));
    // `finish` gives the same bins. A last bin of 5 missing sites is T, as
    // in fq2psmcfa, with the 5 sites past the end callable.
    let mut b = Binner::new(10);
    b.add_missing(95);
    let mut same = b.clone();
    assert_eq!(b.finish(), [2, 2, 2, 2, 2, 2, 2, 2, 2, 0]);
    assert_eq!(finish(&mut same).1.unwrap(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 5]);
    let mut b = Binner::new(MAX_COUNTED_BIN_SIZE);
    b.add_called(MAX_COUNTED_BIN_SIZE as u64 + 1);
    assert_eq!(finish(&mut b), (vec![0, 0], Some(vec![u16::MAX, u16::MAX])));
}

#[test]
fn the_binner_bins_larger_bins_without_counting_them() {
    let mut sizes = vec![MAX_COUNTED_BIN_SIZE + 1, 1 << 30];
    if cfg!(target_pointer_width = "64") {
        sizes.push(1 << 40);
    }
    for s in sizes {
        let s64 = s as u64;
        let mut b = Binner::new(s);
        // At most 90% missing and a heterozygous site: K.
        b.add_missing(9 * s64 / 10);
        b.add_het();
        b.add_called(s64 - 9 * s64 / 10 - 1);
        // All missing: N.
        b.add_missing(s64);
        // A last bin of one called site: T.
        b.add_called(1);
        assert_eq!(finish(&mut b), (vec![1, 2, 0], None), "s {s}");
    }
}

/// Write multihetsep rows and read them for the pair (0, 1).
fn read_mhs_text(text: &str, bin_size: usize, batch_size: Option<usize>) -> Observations {
    let file = TempFile::new("callable_mhs", "multihetsep", text);
    let opts = MhsOptions {
        bin_size,
        batch_size,
        pair: (0, 1),
    };
    read_mhs_with(&file.0, &opts).unwrap()
}

#[test]
fn multihetsep_counts_follow_the_placement_of_uncalled_sites() {
    // 19 uncalled sites, then the heterozygous site at 20; 3 called sites,
    // then the heterozygous site at 24.
    let obs = read_mhs_text("chr1 20 1 AC\nchr1 24 4 AC\n", 4, None);
    assert_eq!(obs.rows, [vec![2, 2, 2, 2, 1, 1]]);
    let c = obs.callable.as_ref().unwrap();
    assert_eq!(c.bin_size(), 4);
    assert_eq!(c.to_counts(), [vec![0, 0, 0, 0, 1, 4]]);

    // Random files against the placement read site by site: the uncalled
    // sites of each interval first, then its called ones, then the row.
    let mut rng = Rng(11);
    for case in 0..300 {
        let s = [1, 3, 10, 100][case % 4];
        let mut text = String::new();
        let mut want_bins = Vec::new();
        let mut want_counts = Vec::new();
        for chrom in 0..rng.range(1, 3) {
            let mut sites = Vec::new();
            let mut pos = 0;
            for _ in 0..rng.range(1, 40) {
                let delta = match rng.range(0, 4) {
                    0 => rng.range(1, 3),
                    1 => rng.range(100, 600),
                    _ => rng.range(1, 80),
                };
                let called = rng.range(1, delta);
                pos += delta;
                // Het, hom and missing for the pair; each row segregates.
                let (alleles, site) = match rng.range(0, 2) {
                    0 => ("ACAA", Site::Het),
                    1 => ("AACA", Site::Called),
                    _ => ("A?CA", Site::Missing),
                };
                writeln!(text, "chr{chrom}\t{pos}\t{called}\t{alleles}").unwrap();
                sites.extend(std::iter::repeat_n(
                    Site::Missing,
                    (delta - called) as usize,
                ));
                sites.extend(std::iter::repeat_n(Site::Called, called as usize - 1));
                sites.push(site);
            }
            let (bins, counts) = reference_bins(&sites, s);
            want_bins.push(bins);
            want_counts.push(counts);
        }
        let obs = read_mhs_text(&text, s, None);
        assert_eq!(obs.rows, want_bins, "case {case}");
        let c = obs.callable.as_ref().unwrap();
        assert_eq!(c.to_counts(), want_counts, "case {case}");
        obs.validate().unwrap();
        // Batched rows cut the counts with the bins.
        let batched = read_mhs_text(&text, s, Some(7));
        let counts = batched.callable.as_ref().unwrap();
        let rejoined: Vec<u16> = counts.to_counts().concat();
        assert_eq!(rejoined, want_counts.concat(), "case {case}");
        assert!(
            counts
                .rows()
                .iter()
                .zip(&batched.rows)
                .all(|(c, r)| c.len() == r.len())
        );
    }
}

/// A one-sample VCF of two chromosomes with `##contig` lengths, and a
/// callable mask that leaves out sites 401-420 and 991-1050 of chr1 and
/// 301-333 of chr2.
fn small_vcf() -> (TempFile, TempFile) {
    let vcf = "##fileformat=VCFv4.2\n\
               ##contig=<ID=chr1,length=1050>\n\
               ##contig=<ID=chr2,length=333>\n\
               #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
               chr1\t10\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n\
               chr1\t500\t.\tA\tG\t50\tPASS\t.\tGT\t0/1\n\
               chr1\t777\t.\tA\tG\t50\tPASS\t.\tGT\t./.\n\
               chr2\t100\t.\tC\tT\t50\tPASS\t.\tGT\t0/1\n";
    let mask = "chr1\t0\t400\nchr1\t420\t990\nchr2\t0\t300\n";
    (
        TempFile::new("callable_small", "vcf", vcf),
        TempFile::new("callable_small_mask", "bed", mask),
    )
}

/// Every kept sequence's counts sum to its called positions plus the
/// positions past its end in its last bin, which count as callable.
fn check_sums(data: &Dataset, bin_size: usize, ctx: &str) {
    let counts = data.sequence_callable().expect(ctx);
    assert_eq!(counts.len(), data.seqs.len(), "{ctx}");
    for (info, c) in data.seqs.iter().zip(&counts) {
        assert_eq!(c.len(), info.n_bins, "{ctx} {}", info.name);
        let padding = (bin_size * info.n_bins) as u64 - info.extent.unwrap();
        let sum: u64 = c.iter().map(u64::from).sum();
        assert_eq!(sum, info.called.unwrap() + padding, "{ctx} {}", info.name);
    }
    data.obs.validate().expect(ctx);
}

#[test]
fn vcf_input_loads_with_its_callable_bases() {
    let (vcf, mask) = small_vcf();
    let opts = |batch_size| InputOptions {
        format: Some(InputFormat::Vcf),
        vcf_mode: VcfMode::Mask,
        masks: vec![mask.0.clone()],
        batch_size,
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let whole = load(&vcf.0, &opts(None)).unwrap();
    let c = whole.sequence_callable().unwrap();
    // chr1: the heterozygous bins 1-100 and 401-500 (20 sites masked), the
    // missing genotype at 777, sites 991-1000 and the last bin, 1001-1050,
    // masked (its 50 sites past the end count as callable).
    let mut chr1 = vec![100u16; 11];
    (chr1[4], chr1[7], chr1[9], chr1[10]) = (80, 99, 90, 50);
    assert_eq!(c[0].to_vec(), chr1);
    assert_eq!(c[1].to_vec(), [100, 100, 100, 67]);
    assert_eq!(whole.seqs[0].called, Some(1050 - 81));
    check_sums(&whole, 100, "whole");

    for batch in [1, 3, 10] {
        let batched = load(&vcf.0, &opts(Some(batch))).unwrap();
        assert_eq!(batched.sequence_callable(), whole.sequence_callable());
        assert!(
            batched
                .obs
                .callable
                .as_ref()
                .unwrap()
                .rows()
                .iter()
                .all(|r| r.len() <= batch)
        );
        check_sums(&batched, 100, &format!("batch {batch}"));
    }

    // read_vcf_with gives the same counts.
    let read = read_vcf_with(
        &vcf.0,
        &VcfOptions {
            masks: vec![mask.0.clone()],
            ..VcfOptions::default()
        },
    )
    .unwrap();
    assert_eq!(read.obs.callable, whole.obs.callable);
}

#[test]
fn every_reader_but_psmcfa_reports_callable_bases() {
    // Consensus mode on the route A all-sites VCF, every sequence kept.
    let opts = InputOptions {
        format: Some(InputFormat::Vcf),
        bin_size: 50,
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let data = load(&fixture("route_a/allsites.vcf.gz"), &opts).unwrap();
    assert_eq!(data.seqs.len(), 3);
    check_sums(&data, 50, "consensus");
    let counts = data.sequence_callable().unwrap();
    assert!(
        counts
            .iter()
            .flat_map(|c| c.iter())
            .any(|c| c > 5 && c < 50),
        "some bins are partly called"
    );

    // Multihetsep.
    let opts = InputOptions {
        format: Some(InputFormat::Mhs),
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let data = load(&fixture("route_a/expected/snps.chr1.multihetsep"), &opts).unwrap();
    check_sums(&data, 100, "multihetsep");

    // psmcfa has none.
    let path = fixture("c_reference/papuan_chr1_3seg.psmcfa");
    let data = load(&path, &InputOptions::default()).unwrap();
    assert!(data.obs.callable.is_none());
    assert!(data.sequence_callable().is_none());
    assert!(read_psmcfa(&path, Some(1000)).unwrap().callable.is_none());
}

#[test]
fn counts_are_kept_when_every_sequence_is_dropped() {
    let (vcf, mask) = small_vcf();
    let opts = InputOptions {
        format: Some(InputFormat::Vcf),
        vcf_mode: VcfMode::Mask,
        masks: vec![mask.0.clone()],
        ..InputOptions::default()
    };
    let data = load(&vcf.0, &opts).unwrap();
    assert!(data.obs.rows.is_empty() && data.seqs.is_empty());
    let c = data.obs.callable.as_ref().unwrap();
    assert_eq!((c.bin_size(), c.rows().len()), (100, 0));
    assert_eq!(data.sequence_callable(), Some(Vec::new()));
}

#[test]
fn bins_too_large_to_count_load_as_before_without_counts() {
    let (vcf, mask) = small_vcf();
    let vcf_opts = |vcf_mode, bin_size| InputOptions {
        format: Some(InputFormat::Vcf),
        vcf_mode,
        bin_size,
        masks: vec![mask.0.clone()],
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let mhs = TempFile::new("callable_big_mhs", "multihetsep", "chr1\t20\t1\tAC\n");
    let mut sizes = vec![MAX_COUNTED_BIN_SIZE + 1, 1 << 30];
    if cfg!(target_pointer_width = "64") {
        // Beyond 32 bits too.
        sizes.push(1 << 33);
    }
    for s in sizes {
        // Each sequence is one bin, heterozygous (in mask mode, 81 of the
        // 1050 positions of chr1 are missing, far from 90% of a bin).
        for mode in [VcfMode::Mask, VcfMode::Consensus, VcfMode::Auto] {
            let ctx = format!("s {s}, {mode:?}");
            let data = load(&vcf.0, &vcf_opts(mode, s)).unwrap_or_else(|e| panic!("{ctx}: {e:#}"));
            assert_eq!(data.seqs.len(), 2, "{ctx}");
            assert!(data.obs.rows.iter().all(|r| r.len() == 1), "{ctx}");
            if mode == VcfMode::Mask {
                assert_eq!(data.obs.rows, [[1], [1]], "{ctx}");
            }
            assert!(data.obs.callable.is_none(), "{ctx}");
            assert!(data.sequence_callable().is_none(), "{ctx}");
        }
        let read = read_vcf_with(
            &vcf.0,
            &VcfOptions {
                bin_size: s,
                masks: vec![mask.0.clone()],
                ..VcfOptions::default()
            },
        )
        .unwrap();
        assert_eq!(read.obs.rows, [[1], [1]], "s {s}");
        assert!(read.obs.callable.is_none(), "s {s}");

        let mhs_opts = InputOptions {
            format: Some(InputFormat::Mhs),
            bin_size: s,
            filter: SeqFilter::DISABLED,
            ..InputOptions::default()
        };
        let data = load(&mhs.0, &mhs_opts).unwrap();
        assert_eq!(data.obs.rows, [[1]], "s {s}");
        assert!(data.obs.callable.is_none(), "s {s}");
        let opts = MhsOptions {
            bin_size: s,
            ..MhsOptions::default()
        };
        let obs = read_mhs_with(&mhs.0, &opts).unwrap();
        assert_eq!(obs.rows, [[1]], "s {s}");
        assert!(obs.callable.is_none(), "s {s}");
    }

    // The largest counted bin size has counts.
    let data = load(&vcf.0, &vcf_opts(VcfMode::Mask, MAX_COUNTED_BIN_SIZE)).unwrap();
    check_sums(&data, MAX_COUNTED_BIN_SIZE, "largest counted bins");

    // psmcfa input is already binned: its bin size only scales the results.
    let path = fixture("c_reference/papuan_chr1_3seg.psmcfa");
    let opts = InputOptions {
        bin_size: MAX_COUNTED_BIN_SIZE + 1,
        ..InputOptions::default()
    };
    assert!(load(&path, &opts).unwrap().obs.callable.is_none());
}

/// A one-sample VCF of one chromosome of `len` positions with a
/// heterozygous SNP every 1000 positions, and a callable mask of the
/// intervals `callable` (0-based, half-open).
fn masked_vcf(len: u64, callable: &[(u64, u64)]) -> (TempFile, TempFile) {
    let mut vcf = format!(
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length={len}>\n\
         #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
    );
    for pos in (500..=len).step_by(1000) {
        writeln!(vcf, "chr1\t{pos}\t.\tA\tC\t50\tPASS\t.\tGT\t0/1").unwrap();
    }
    let mut bed = String::new();
    for (start, end) in callable {
        writeln!(bed, "chr1\t{start}\t{end}").unwrap();
    }
    (
        TempFile::new("callable_memory", "vcf", &vcf),
        TempFile::new("callable_memory_mask", "bed", &bed),
    )
}

#[test]
fn counts_take_no_memory_for_full_bins_and_little_for_runs_of_missing_data() {
    // 1 Mb in bins of 100: 10,000 bins.
    let len = 1_000_000;
    let n_bins = 10_000;
    let load_masked = |callable: &[(u64, u64)], batch_size| {
        let (vcf, mask) = masked_vcf(len, callable);
        let opts = InputOptions {
            format: Some(InputFormat::Vcf),
            vcf_mode: VcfMode::Mask,
            masks: vec![mask.0.clone()],
            batch_size,
            filter: SeqFilter::DISABLED,
            ..InputOptions::default()
        };
        load(&vcf.0, &opts).unwrap()
    };
    // The bytes the counts of the bins take, besides one struct per row.
    let counts_bytes = |data: &Dataset| {
        let c = data.obs.callable.as_ref().unwrap();
        c.rows().iter().map(|r| r.heap_bytes()).sum::<usize>()
    };
    for batch_size in [None, Some(1000)] {
        let ctx = format!("batch {batch_size:?}");
        // Complete data: every bin full, no memory.
        let data = load_masked(&[(0, len)], batch_size);
        assert_eq!(data.n_bins(), n_bins);
        assert_eq!(counts_bytes(&data), 0, "{ctx}");

        // A run of 300 missing bases every 10 kb (positions 51-350,
        // 10051-10350, ...): a bin of 50 callable bases, two empty bins and
        // another of 50, so three runs of equal counts at 10 bytes each
        // (none crosses a row of 1000 bins).
        let mut callable = vec![(0, 50)];
        callable
            .extend((0..len / 10_000).map(|i| (i * 10_000 + 350, (i * 10_000 + 10_050).min(len))));
        let data = load_masked(&callable, batch_size);
        let counts = data.sequence_callable().unwrap();
        let mut gap = [100u16; 100];
        gap[..4].copy_from_slice(&[50, 0, 0, 50]);
        assert_eq!(counts[0].to_vec(), gap.repeat(100), "{ctx}");
        assert_eq!(counts_bytes(&data), 100 * 3 * 10, "{ctx}");
        check_sums(&data, 100, &ctx);

        // One or two missing bases in every bin, in turn: every bin partly
        // called and different from the last, so one byte per bin.
        let callable: Vec<(u64, u64)> = (0..n_bins as u64)
            .map(|i| (i * 100 + 1 + i % 2, i * 100 + 100))
            .collect();
        let data = load_masked(&callable, batch_size);
        let counts = data.sequence_callable().unwrap();
        assert!(
            counts[0]
                .iter()
                .enumerate()
                .all(|(i, c)| c == 99 - (i % 2) as u16),
            "{ctx}"
        );
        assert_eq!(counts_bytes(&data), n_bins, "{ctx}");
    }
}

#[test]
fn psmcfa_written_from_counted_input_holds_the_same_bins() {
    let (vcf, mask) = small_vcf();
    let opts = InputOptions {
        format: Some(InputFormat::Vcf),
        vcf_mode: VcfMode::Mask,
        masks: vec![mask.0.clone()],
        batch_size: Some(4),
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let data = load(&vcf.0, &opts).unwrap();
    assert!(data.obs.callable.is_some());
    let mut without = data.clone();
    without.obs.callable = None;
    let out = TempFile::new("callable_written", "psmcfa", "");
    write_psmcfa(&out.0, &data).unwrap();
    let with_counts = fs::read(&out.0).unwrap();
    write_psmcfa(&out.0, &without).unwrap();
    assert_eq!(with_counts, fs::read(&out.0).unwrap());
    let back = read_psmcfa(&out.0, Some(4)).unwrap();
    assert_eq!(back.rows, data.obs.rows);
    assert!(back.callable.is_none());
}
