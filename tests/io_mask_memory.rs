//! Callable masks are kept as merged intervals, not row by row: a mask with
//! one row per callable site (what an all-sites VCF gives, billions of rows
//! for a human genome) must not need memory in proportion to its rows.
//!
//! The test counts the bytes allocated through a global allocator, so this
//! file holds a single test: tests of one binary run in parallel threads,
//! and their allocations would mix.

mod common;

use common::TempDir;
use psmc_rs::io::vcf::{VcfOptions, read_vcf_with};
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The system allocator, counting the bytes in use and their peak.
struct Counting;

static IN_USE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(n: usize) {
    let now = IN_USE.fetch_add(n, Ordering::SeqCst) + n;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

fn shrank(n: usize) {
    IN_USE.fetch_sub(n, Ordering::SeqCst);
}

// SAFETY: every call is forwarded to `System` with the same arguments.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        shrank(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                grew(new_size - layout.size());
            } else {
                shrank(layout.size() - new_size);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Peak bytes allocated by `f` beyond what was in use when it started.
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = IN_USE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let out = f();
    (out, PEAK.load(Ordering::SeqCst) - base)
}

/// A BED mask with one row per site for positions `1..=sites` of chr1, in
/// increasing or, with `reversed`, decreasing order.
fn per_site_mask(path: &Path, sites: u64, reversed: bool) {
    let mut text = String::with_capacity(sites as usize * 20);
    let mut row = |i: u64| writeln!(text, "chr1\t{i}\t{}", i + 1).unwrap();
    if reversed {
        (0..sites).rev().for_each(&mut row);
    } else {
        (0..sites).for_each(&mut row);
    }
    fs::write(path, text).unwrap();
}

#[test]
fn per_site_masks_take_memory_in_proportion_to_merged_intervals() {
    const LEN: u64 = 2_000_000;
    const SITES: u64 = 1_000_000;
    let dir = TempDir::new("io_mask_memory");
    let vcf = dir.path("calls.vcf");
    fs::write(
        &vcf,
        format!(
            "##fileformat=VCFv4.2\n##contig=<ID=chr1,length={LEN}>\n\
             #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
             chr1\t1000\t.\tA\tG\t50\tPASS\t.\tGT\t0/1\n\
             chr1\t500000\t.\tA\tG\t50\tPASS\t.\tGT\t0/1\n"
        ),
    )
    .unwrap();
    let mask = dir.path("mask.bed");
    for reversed in [false, true] {
        per_site_mask(&mask, SITES, reversed);
        let opts = VcfOptions {
            masks: vec![mask.clone()],
            ..VcfOptions::default()
        };
        let (read, peak) = peak_of(|| read_vcf_with(&vcf, &opts).unwrap());
        // The mask is one interval, [1, SITES]: the rest is missing.
        assert_eq!(read.summary.missing_sites, LEN - SITES);
        assert_eq!(read.summary.het_snps, 2);
        // Holding the rows would take 16 bytes each (16 MB), and more while
        // the list grows and is merged; the merged mask is a few bytes.
        assert!(
            peak < 4 << 20,
            "reading a mask of {SITES} per-site rows (reversed: {reversed}) allocated \
             {peak} bytes at its peak"
        );
    }
}
