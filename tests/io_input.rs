//! The unified loader: format and VCF mode detection, the fq2psmcfa output
//! filter, per-sequence information and heterozygous sites.

mod common;

use common::{TempDir, VCF_HEADER, gzip, mask_summary, vcf_summary};
use psmc_rs::io::input::{
    DropReason, HetSite, InputFormat, InputOptions, MODE_DETECTION_RECORDS, SeqFilter,
    SourceSummary, VcfDetails, VcfMode, detect_vcf_mode, load,
};
use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::io::vcf::read_vcf;
use psmc_rs::io::write::write_psmcfa_to;
use std::fmt::Write as _;
use std::path::Path;

/// An all-sites VCF as `bcftools call -c` writes it, with a heterozygous
/// SNP every `het_every` positions.
fn call_c_vcf(chrom: &str, len: u64, het_every: u64, depth: u64) -> String {
    let mut out = String::from(VCF_HEADER);
    for pos in 1..=len {
        let (alt, af1, fq, gt) = if pos % het_every == 0 {
            ("G", "0.5", "40", "0/1")
        } else {
            (".", "0", "-60", "0/0")
        };
        writeln!(
            out,
            "{chrom}\t{pos}\t.\tA\t{alt}\t50\t.\tDP={depth};MQ=50;AF1={af1};FQ={fq}\tGT\t{gt}"
        )
        .unwrap();
    }
    out
}

/// A VCF read in `mode`.
fn vcf_opts(mode: VcfMode) -> InputOptions {
    InputOptions {
        format: Some(InputFormat::Vcf),
        vcf_mode: mode,
        ..InputOptions::default()
    }
}

/// [`vcf_opts`] without the output filter.
fn unfiltered(mode: VcfMode) -> InputOptions {
    InputOptions {
        filter: SeqFilter::DISABLED,
        ..vcf_opts(mode)
    }
}

/// The defaults, with bins of `bin_size` and without the output filter.
fn binned(bin_size: usize) -> InputOptions {
    InputOptions {
        bin_size,
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    }
}

/// The message of the error loading `path` with `opts`.
fn error(path: &Path, opts: &InputOptions) -> String {
    format!("{:#}", load(path, opts).unwrap_err())
}

#[test]
fn formats_are_recognised_from_the_content() {
    let dir = TempDir::new("io_input");
    let psmcfa = dir.write("psmcfa.txt", "\n>chr1\nTTKN\n");
    for (path, format) in [
        (
            dir.write("vcf.txt", call_c_vcf("c", 3, 2, 10)),
            InputFormat::Vcf,
        ),
        (psmcfa.clone(), InputFormat::Psmcfa),
        (dir.write("mhs.txt", "chr1\t10\t10\tAC\n"), InputFormat::Mhs),
    ] {
        assert_eq!(InputFormat::sniff(&path).unwrap(), format);
    }
    assert!(InputFormat::sniff(&dir.write("empty.txt", "\n\n")).is_err());

    let data = load(&psmcfa, &InputOptions::default()).unwrap();
    assert_eq!(data.source.format(), InputFormat::Psmcfa);
    assert_eq!(data.obs.rows, [[0, 0, 1, 2]]);
}

#[test]
fn bcf_input_is_refused_with_a_conversion_hint() {
    // BCF 2.2 as bcftools writes it: magic, header length, header text,
    // then binary records. -Ob compresses it (gzip framing here).
    let mut bcf = b"BCF\x02\x02".to_vec();
    let header = format!("{VCF_HEADER}\0");
    bcf.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bcf.extend_from_slice(header.as_bytes());
    bcf.extend_from_slice(&[0x1c, 0, 0, 0, 0xff, 0xfe, 0x80, 0x7f]);
    let gz = gzip(&bcf);
    let dir = TempDir::new("io_input");
    for (bytes, format) in [
        (&bcf, None),
        (&gz, None),
        (&gz, Some(InputFormat::Vcf)),
        (&gz, Some(InputFormat::Mhs)),
    ] {
        let opts = InputOptions {
            format,
            ..InputOptions::default()
        };
        let msg = error(&dir.write("in.bcf", bytes), &opts);
        assert!(
            msg.contains("is a BCF file") && msg.contains("bcftools view -Oz"),
            "{format:?}: {msg}"
        );
    }
}

#[test]
fn a_utf8_byte_order_mark_is_skipped() {
    const BOM: &str = "\u{feff}";
    let dir = TempDir::new("io_input");
    let psmcfa = dir.write("in.psmcfa", format!("{BOM}>chr1\nTTKN\n"));
    let data = load(&psmcfa, &InputOptions::default()).unwrap();
    assert_eq!(data.source.format(), InputFormat::Psmcfa);
    assert_eq!(data.seqs[0].name, "chr1");
    assert_eq!(data.obs.rows, [[0, 0, 1, 2]]);

    let mhs = dir.write("in.mhs", format!("{BOM}chr1\t10\t10\tAC\n"));
    let data = load(&mhs, &binned(5)).unwrap();
    assert_eq!(data.source.format(), InputFormat::Mhs);
    assert_eq!(data.seqs[0].name, "chr1");

    let vcf = dir.write("in.vcf", format!("{BOM}{}", call_c_vcf("c", 30, 3, 20)));
    let data = load(&vcf, &binned(100)).unwrap();
    assert_eq!(data.source.format(), InputFormat::Vcf);
    assert_eq!(data.seqs[0].het_sites, Some(10));
}

#[test]
fn a_failed_multihetsep_guess_says_how_the_format_was_chosen() {
    let dir = TempDir::new("io_input");
    let notes = dir.write("notes.txt", "Some notes, not data\n");
    let err = error(&notes, &InputOptions::default());
    assert!(
        err.contains("failed to read multihetsep")
            && err.contains("recognised from the first line"),
        "{err}"
    );
    let forced = InputOptions {
        format: Some(InputFormat::Mhs),
        ..InputOptions::default()
    };
    let err = error(&notes, &forced);
    assert!(!err.contains("recognised from the first line"), "{err}");
    let guessed = |text: &str| error(&dir.write("in.txt", text), &InputOptions::default());
    // The note names the option, and comes with a row that lacks columns
    // anywhere in the file.
    let err = guessed("chr1\t100\t50\tAC\nchr1 200\n");
    assert!(
        err.contains("recognised from the first line") && err.contains("give --input-format"),
        "{err}"
    );
    assert!(err.contains("row 2: missing nr_called"), "{err}");
    // An error in a later row of a file that started as multihetsep is
    // that file's error, without the note.
    for (text, error) in [
        (
            "chr1\t100\t50\tAC\nchr2\t90\t10\tAC\nchr1\t300\t100\tAG\n",
            "chromosome 'chr1' appears again",
        ),
        ("chr1\t100\t50\tAC\nchr1\t90\t10\tAC\n", "pos must increase"),
    ] {
        let err = guessed(text);
        assert!(err.contains(error), "{err}");
        assert!(!err.contains("recognised from the first line"), "{err}");
    }
}

#[test]
fn vcf_headers_need_not_start_with_fileformat() {
    let dir = TempDir::new("io_input");
    let body = call_c_vcf("c", 40, 4, 20);
    let records: String = body.lines().skip(2).map(|l| format!("{l}\n")).collect();
    let chrom_line = VCF_HEADER.lines().nth(1).unwrap();
    let opts = binned(10);
    let as_vcf = InputOptions {
        format: Some(InputFormat::Vcf),
        ..opts.clone()
    };
    let expected = load(&dir.write("ref.vcf", &body), &as_vcf).unwrap();
    assert_eq!(expected.obs.rows, [vec![1; 4]]);
    // vcf2fq skips every '#' line and needs no header at all.
    for text in [
        format!("##contig=<ID=c,length=4000>\n{chrom_line}\n{records}"),
        format!("##source=bcftools\n##fileformat=VCFv4.2\n{chrom_line}\n{records}"),
        format!("\n{chrom_line}\n{records}"),
        format!("# a comment\n{chrom_line}\n{records}"),
        records.clone(),
    ] {
        let vcf = dir.write("in.vcf", &text);
        assert_eq!(
            InputFormat::sniff(&vcf).unwrap(),
            InputFormat::Vcf,
            "{text}"
        );
        let data = load(&vcf, &opts).unwrap();
        assert_eq!(vcf_summary(&data).mode.mode, VcfMode::Consensus, "{text}");
        assert_eq!(data.obs.rows, expected.obs.rows);
    }

    // Comment lines before multihetsep or psmcfa content are skipped.
    let mhs = dir.write("mhs.txt", "# made by hand\nchr1\t10\t10\tAC\n");
    for (path, format) in [
        (mhs.clone(), InputFormat::Mhs),
        (
            dir.write("psmcfa.txt", "# made by hand\n>chr1\nTTKN\n"),
            InputFormat::Psmcfa,
        ),
        (
            dir.write("comments.txt", "# nothing else\n\n"),
            InputFormat::Mhs,
        ),
    ] {
        assert_eq!(InputFormat::sniff(&path).unwrap(), format);
    }
    assert_eq!(load(&mhs, &opts).unwrap().source.format(), InputFormat::Mhs);
}

#[test]
fn auto_mode_reads_call_c_output_as_consensus() {
    let dir = TempDir::new("io_input");
    let vcf = dir.write("in.vcf", call_c_vcf("c", 200, 10, 20));
    let choice = detect_vcf_mode(&vcf).unwrap();
    assert_eq!((choice.mode, choice.forced), (VcfMode::Consensus, false));
    assert_eq!(
        (choice.inspected, choice.alt_dot, choice.alt_dot_with_fq),
        (200, 180, 180)
    );
    assert!(
        choice.reason.contains("bcftools call -c"),
        "{}",
        choice.reason
    );

    let data = load(&vcf, &binned(10)).unwrap();
    let v = vcf_summary(&data);
    assert_eq!(v.mode.mode, VcfMode::Consensus);
    assert!(matches!(v.details, VcfDetails::Consensus(_)));
    assert_eq!(data.obs.rows, [vec![1; 20]]);
}

#[test]
fn auto_mode_reads_variants_only_vcfs_with_masks() {
    let dir = TempDir::new("io_input");
    let vcf = dir.write("in.vcf",
        format!(
            "{VCF_HEADER}c\t5\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\nc\t9\t.\tA\t.\t50\tPASS\tFQ=-3\tGT\t0/0\n\
             c\t12\t.\tA\tG\t50\tPASS\t.\tGT\t1/1\n"
        ),
    );
    let choice = detect_vcf_mode(&vcf).unwrap();
    assert_eq!(choice.mode, VcfMode::Mask);
    assert_eq!((choice.inspected, choice.alt_dot), (3, 1));
    assert!(
        choice.reason.contains("VCF of variant sites"),
        "{}",
        choice.reason
    );

    let data = load(&vcf, &binned(4)).unwrap();
    // Identical to the existing mask-mode reader.
    let expected = read_vcf(&vcf, None, 4, None, &[], &[]).unwrap();
    assert_eq!(data.obs.rows, expected.rows);
    assert_eq!(mask_summary(&data).het_snps, 1);
    // Without a mask, the mask-mode warning is passed on.
    assert!(data.warnings()[0].contains("no callable mask"));
}

#[test]
fn auto_mode_names_a_gvcf_as_such() {
    // DeepVariant: reference and no-call blocks with END, and variants
    // with a symbolic <*> allele.
    let dir = TempDir::new("io_input");
    let vcf = dir.write(
        "in.vcf",
        format!(
            "{VCF_HEADER}c\t1\t.\tA\t<*>\t0\t.\tEND=40\tGT:GQ\t0/0:50\n\
             c\t41\t.\tA\tC,<*>\t50\tPASS\t.\tGT:GQ\t0/1:40\n\
             c\t42\t.\tA\t<*>\t0\t.\tEND=80\tGT:GQ\t./.:0\n\
             c\t81\t.\tA\t<*>\t0\t.\tEND=100\tGT:GQ\t0/0:50\n"
        ),
    );
    let choice = detect_vcf_mode(&vcf).unwrap();
    assert_eq!((choice.mode, choice.gvcf_blocks), (VcfMode::Mask, 3));
    assert!(
        choice.reason.contains(
            "3 of the first 4 records are reference or no-call blocks with INFO/END, as in a gVCF"
        ) && !choice.reason.contains("variant sites"),
        "{}",
        choice.reason
    );
    let data = load(&vcf, &binned(10)).unwrap();
    assert_eq!(data.seqs[0].called, Some(61));
    assert!(data.warnings().iter().any(|w| w.contains("is a gVCF")));
}

/// `bcftools call -m -g` writes its reference blocks with ALT '.' and
/// nothing at all where there are no reads. Such a file is a gVCF, read in
/// mask mode whether detected or requested, and the positions no record
/// covers are missing even where the callable mask allows them.
#[test]
fn a_bcftools_gvcf_is_read_in_mask_mode_with_its_gaps_missing() {
    let dir = TempDir::new("io_input");
    let vcf = dir.write(
        "in.vcf",
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n\
         #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
         chr1\t1\t.\tG\t.\t.\t.\tEND=298;MIN_DP=1\tGT:DP\t0/0:1\n\
         chr1\t299\t.\tA\tG\t50\t.\tDP=5\tGT:DP\t0/1:5\n\
         chr1\t600\t.\tA\t.\t.\t.\tEND=899;MIN_DP=1\tGT:DP\t0/0:1\n",
    );
    let mask = dir.write("mask.bed", "chr1\t0\t1000\n");
    let choice = detect_vcf_mode(&vcf).unwrap();
    assert_eq!(
        (choice.mode, choice.gvcf_blocks, choice.alt_dot),
        (VcfMode::Mask, 2, 2)
    );
    assert!(choice.reason.contains("as in a gVCF"), "{}", choice.reason);
    for mode in [VcfMode::Auto, VcfMode::Mask] {
        let opts = InputOptions {
            bin_size: 10,
            masks: vec![mask.clone()],
            ..unfiltered(mode)
        };
        let data = load(&vcf, &opts).unwrap();
        // 300..=599 and 900..=1000 have no record.
        let s = &data.seqs[0];
        assert_eq!((s.extent, s.called), (Some(1000), Some(599)), "{mode:?}");
        let m = mask_summary(&data);
        assert_eq!((m.uncovered_sites, m.missing_sites), (401, 401), "{mode:?}");
        let warnings = data.warnings();
        assert!(
            warnings.iter().any(|w| w.contains("is a gVCF")
                && w.contains("so are the positions that no record or block covers (401 here")),
            "{warnings:?}"
        );
    }
    // Without a mask, the gVCF warning stands alone: the generic one would
    // say that positions without a record are called, which is not so here.
    let opts = InputOptions {
        bin_size: 10,
        ..unfiltered(VcfMode::Auto)
    };
    let warnings = load(&vcf, &opts).unwrap().warnings();
    assert!(
        warnings.iter().any(|w| w.contains("is a gVCF")),
        "{warnings:?}"
    );
    assert!(
        !warnings
            .iter()
            .any(|w| w.contains("every position without a VCF record counts")),
        "{warnings:?}"
    );
}

#[test]
fn an_all_sites_vcf_without_fq_is_an_error_with_guidance() {
    // `bcftools call -m` without -v: ALT '.' everywhere, no FQ.
    let mut text = String::from(VCF_HEADER);
    for pos in 1..=100 {
        writeln!(text, "c\t{pos}\t.\tA\t.\t50\t.\tDP=10;MQ=50\tGT\t0/0").unwrap();
    }
    let dir = TempDir::new("io_input");
    let vcf = dir.write("in.vcf", &text);
    let msg = format!("{:#}", detect_vcf_mode(&vcf).unwrap_err());
    for part in ["all-sites VCF", "bcftools call -c", "callable mask"] {
        assert!(msg.contains(part), "{msg}");
    }
    assert!(load(&vcf, &vcf_opts(VcfMode::Auto)).is_err());
    // Forcing mask mode reads the genotypes.
    let data = load(&vcf, &unfiltered(VcfMode::Mask)).unwrap();
    assert_eq!(data.seqs[0].called, Some(100));
}

#[test]
fn mode_detection_thresholds() {
    let line = |pos: u64, alt: &str, info: &str| {
        format!("c\t{pos}\t.\tA\t{alt}\t50\t.\t{info}\tGT\t0/0\n")
    };
    // Records at consecutive positions: `n_dot` with ALT '.' (the first
    // `n_dot_fq` of them with FQ), then `n_var` with ALT C and `var_info`.
    let build = |n_dot: u64, n_dot_fq: u64, n_var: u64, var_info: &str| {
        let mut s = String::from(VCF_HEADER);
        let mut pos = 0;
        for i in 0..n_dot {
            pos += 1;
            let info = if i < n_dot_fq { "DP=5;FQ=-30" } else { "DP=5" };
            s.push_str(&line(pos, ".", info));
        }
        for _ in 0..n_var {
            pos += 1;
            s.push_str(&line(pos, "C", var_info));
        }
        s
    };
    let dir = TempDir::new("io_input");
    let detect = |text: &str| detect_vcf_mode(&dir.write("in.vcf", text));
    // Exactly 50% ALT '.' with exactly 90% FQ: consensus.
    let choice = detect(&build(10, 9, 10, "DP=5")).unwrap();
    assert_eq!(choice.mode, VcfMode::Consensus);
    assert_eq!(
        (
            choice.alt_dot,
            choice.alt_dot_with_fq,
            choice.records_with_fq
        ),
        (10, 9, 9)
    );
    // 89% FQ: error.
    assert!(detect(&build(100, 89, 100, "DP=5")).is_err());
    // Just under 50% ALT '.', without FQ on the other records: mask.
    let choice = detect(&build(9, 9, 10, "DP=5")).unwrap();
    assert_eq!((choice.mode, choice.span), (VcfMode::Mask, 19));
    assert!(choice.reason.contains("INFO/FQ"), "{}", choice.reason);
    // No records at all: mask.
    let choice = detect(VCF_HEADER).unwrap();
    assert_eq!((choice.mode, choice.inspected), (VcfMode::Mask, 0));
    // Only the first records are inspected.
    let n = MODE_DETECTION_RECORDS;
    let mut s = build(0, 0, n, "DP=5");
    for pos in n + 1..=3 * n {
        s.push_str(&line(pos, ".", "DP=5;FQ=-30"));
    }
    let choice = detect(&s).unwrap();
    assert_eq!((choice.mode, choice.inspected), (VcfMode::Mask, n));
}

#[test]
fn dense_call_c_vcfs_with_alt_alleles_at_reference_sites_are_consensus() {
    // `bcftools call -c -A` at high depth: most reference calls (FQ < 0)
    // have an ALT allele, and every record carries FQ.
    let line = |chrom: &str, pos: u64, alt: &str, fq: Option<&str>| {
        let fq = fq.map_or(String::new(), |q| format!(";FQ={q}"));
        format!("{chrom}\t{pos}\t.\tA\t{alt}\t50\t.\tDP=30;MQ=50;AF1=0{fq}\tGT\t0/0\n")
    };
    // `n` records, one every `step` positions from `start`, a third with ALT
    // '.', and FQ on all but `no_fq` of them.
    let build = |chrom: &str, start: u64, n: u64, step: u64, no_fq: u64| {
        (0..n)
            .map(|i| {
                let alt = if i % 3 == 0 { "." } else { "G" };
                let fq = (i >= no_fq).then_some("-40");
                line(chrom, start + i * step, alt, fq)
            })
            .collect::<String>()
    };
    let dir = TempDir::new("io_input");
    let detect = |records: String| {
        detect_vcf_mode(&dir.write("in.vcf", format!("{VCF_HEADER}{records}"))).unwrap()
    };

    let choice = detect(build("c", 1, 300, 1, 0));
    assert_eq!(choice.mode, VcfMode::Consensus);
    assert_eq!(
        (
            choice.inspected,
            choice.alt_dot,
            choice.records_with_fq,
            choice.span
        ),
        (300, 100, 300, 300)
    );
    assert!(choice.reason.contains("call -c -A"), "{}", choice.reason);
    // Exactly 90% with FQ is enough, 89% is not.
    assert_eq!(detect(build("c", 1, 300, 1, 30)).mode, VcfMode::Consensus);
    let choice = detect(build("c", 1, 300, 1, 33));
    assert_eq!(choice.mode, VcfMode::Mask);
    assert!(choice.reason.contains("INFO/FQ"), "{}", choice.reason);
    // Records at half of the positions they span are enough, fewer are not:
    // 300 records over 599, 600 and 601 positions.
    assert_eq!(detect(build("c", 1, 300, 2, 0)).mode, VcfMode::Consensus);
    let with_last = |last: u64| build("c", 1, 299, 2, 0) + &line("c", last, "G", Some("-40"));
    let choice = detect(with_last(600));
    assert_eq!((choice.mode, choice.span), (VcfMode::Consensus, 600));
    let choice = detect(with_last(601));
    assert_eq!((choice.mode, choice.span), (VcfMode::Mask, 601));
    assert!(choice.reason.contains("variant sites"), "{}", choice.reason);
    // The span is counted per chromosome.
    let choice = detect(build("a", 1_000_001, 150, 1, 0) + &build("b", 1, 150, 1, 0));
    assert_eq!((choice.mode, choice.span), (VcfMode::Consensus, 300));
    // A variants-only `call -cv` VCF carries FQ too, but is sparse.
    assert_eq!(detect(build("c", 1, 300, 500, 0)).mode, VcfMode::Mask);
}

#[test]
fn forced_modes_are_reported() {
    let dir = TempDir::new("io_input");
    let vcf = dir.write("in.vcf", call_c_vcf("c", 50, 5, 20));
    let data = load(&vcf, &unfiltered(VcfMode::Mask)).unwrap();
    let mode = &vcf_summary(&data).mode;
    assert_eq!((mode.mode, mode.forced), (VcfMode::Mask, true));
    // Mask mode inspects the first records for gVCF blocks.
    assert_eq!((mode.inspected, mode.gvcf_blocks), (50, 0));
    assert!(mode.reason.contains("requested"));

    // Consensus mode reads INFO only, but a sample the header lacks is
    // still an error.
    let opts = InputOptions {
        sample: Some("nobody".to_string()),
        ..vcf_opts(VcfMode::Consensus)
    };
    let err = error(&vcf, &opts);
    assert!(err.contains("'nobody' not found"), "{err}");
}

#[test]
fn consensus_mode_refuses_a_vcf_with_several_samples() {
    // A joint `bcftools call -c` of two samples: INFO pools them.
    let two: String = call_c_vcf("c", 50, 5, 20)
        .lines()
        .map(|l| match l {
            l if l.starts_with("##") => format!("{l}\n"),
            l if l.starts_with('#') => format!("{l}\ts2\n"),
            l => format!("{l}\t0/0\n"),
        })
        .collect();
    let dir = TempDir::new("io_input");
    let vcf = dir.write("two.vcf", &two);
    let sample = |name: &str| Some(name.to_string());
    let refused = |opts: &InputOptions| {
        let msg = error(&vcf, opts);
        assert!(
            msg.contains("2 samples (s1, s2)")
                && msg.contains("pooled")
                && msg.contains("--vcf-mode mask --vcf-sample"),
            "{msg}"
        );
        msg
    };
    // Detected or requested, with or without a sample name.
    refused(&InputOptions::default());
    refused(&vcf_opts(VcfMode::Consensus));
    let msg = refused(&InputOptions {
        sample: sample("s2"),
        ..InputOptions::default()
    });
    assert!(msg.contains("cannot read sample 's2' alone"), "{msg}");
    // A name the header lacks is still reported as such.
    let opts = InputOptions {
        sample: sample("nobody"),
        ..InputOptions::default()
    };
    let err = error(&vcf, &opts);
    assert!(err.contains("'nobody' not found"), "{err}");

    // Mask mode reads the genotypes of the sample asked for.
    let opts = InputOptions {
        sample: sample("s2"),
        ..unfiltered(VcfMode::Mask)
    };
    assert_eq!(load(&vcf, &opts).unwrap().het_bins(), 0);
    // So does consensus mode for a one-sample VCF named on the command line.
    let one = dir.write("one.vcf", call_c_vcf("c", 50, 5, 20));
    let opts = InputOptions {
        sample: sample("s1"),
        ..binned(5)
    };
    assert_eq!(load(&one, &opts).unwrap().het_bins(), 10);
}

#[test]
fn the_output_filter_follows_fq2psmcfa() {
    let f = SeqFilter::default();
    assert_eq!((f.min_good_bases, f.min_good_fraction), (10_000, 0.2));
    let dropped = |few_good_bases, low_good_fraction| {
        Some(DropReason::Filter {
            few_good_bases,
            low_good_fraction,
        })
    };
    assert_eq!(f.verdict(10_000, 50_000), None);
    assert_eq!(f.verdict(9_999, 20_000), dropped(true, false));
    assert_eq!(f.verdict(10_000, 50_001), dropped(false, true));
    assert_eq!(f.verdict(10, 1_000_000), dropped(true, true));
    assert_eq!(SeqFilter::DISABLED.verdict(0, 1), None);
    let only_fraction = SeqFilter {
        min_good_bases: 0,
        ..SeqFilter::default()
    };
    assert_eq!(only_fraction.verdict(1, 5), None);
    assert!(only_fraction.verdict(1, 6).is_some());
}

#[test]
fn the_output_filter_applies_to_every_vcf_mode() {
    // c1: 20,000 bases, all called. c2: 5,000 called bases only.
    let mut text = call_c_vcf("c1", 20_000, 100, 20);
    text.push_str(&call_c_vcf("c2", 5_000, 100, 20).replace(VCF_HEADER, ""));
    let dir = TempDir::new("io_input");
    let vcf = dir.write("in.vcf", &text);
    for mode in [VcfMode::Consensus, VcfMode::Mask] {
        let data = load(&vcf, &vcf_opts(mode)).unwrap();
        let names: Vec<_> = data.seqs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["c1"], "{mode:?}");
        assert_eq!((data.dropped.len(), data.obs.rows.len()), (1, 1));
        let d = &data.dropped[0];
        assert_eq!(
            (d.name.as_str(), d.extent, d.good),
            ("c2", Some(5_000), Some(5_000))
        );
        assert_eq!(
            d.reason,
            DropReason::Filter {
                few_good_bases: true,
                low_good_fraction: false
            }
        );
        let w = data.warnings();
        assert!(
            w.iter()
                .any(|w| w.contains("1 sequence dropped") && w.contains("c2")),
            "{w:?}"
        );
        // Disabled, both are kept.
        let all = load(&vcf, &unfiltered(mode)).unwrap();
        assert_eq!(all.seqs.len(), 2);
        assert!(all.dropped.is_empty());
    }
}

#[test]
fn mask_mode_good_bases_are_the_callable_sites() {
    // A 30,000 bp contig: the good bases are the callable positions, and
    // the sequence extends to the ##contig length.
    let dir = TempDir::new("io_input");
    let vcf = dir.write(
        "in.vcf",
        "##fileformat=VCFv4.2\n##contig=<ID=c,length=30000>\n\
         #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
         c\t100\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n",
    );
    // 10,000 callable positions: enough, and a third of the sequence.
    let opts = InputOptions {
        masks: vec![dir.write("third.bed", "c\t0\t10000\n")],
        ..vcf_opts(VcfMode::Mask)
    };
    let data = load(&vcf, &opts).unwrap();
    assert_eq!(data.seqs.len(), 1);
    let s = &data.seqs[0];
    assert_eq!(
        (s.extent, s.called, s.het_sites, s.n_bins),
        (Some(30_000), Some(10_000), Some(1), 300)
    );

    // 5,999 callable positions: just under a fifth.
    let opts = InputOptions {
        masks: vec![dir.write("fifth.bed", "c\t0\t5999\n")],
        filter: SeqFilter {
            min_good_bases: 0,
            ..SeqFilter::default()
        },
        ..vcf_opts(VcfMode::Mask)
    };
    let data = load(&vcf, &opts).unwrap();
    assert!(data.seqs.is_empty());
    assert_eq!(
        data.dropped[0].reason,
        DropReason::Filter {
            few_good_bases: false,
            low_good_fraction: true
        }
    );
    // Nothing left: the loader succeeds, ensure_usable explains.
    assert!(data.obs.rows.is_empty());
    let msg = format!("{:#}", data.ensure_usable().unwrap_err());
    assert!(
        msg.contains("no sequence of the vcf input is left"),
        "{msg}"
    );
    assert!(msg.contains("c: 30,000 bases, 5,999 called"), "{msg}");
}

#[test]
fn multihetsep_input_is_filtered_and_described() {
    let dir = TempDir::new("io_input");
    let two = dir.write(
        "two.mhs",
        "A\t20000\t19000\tAC\nA\t20010\t10\tAA,CC\nA\t20020\t10\tAC,CA\nB\t100\t50\tAC\n",
    );
    let opts = InputOptions {
        format: Some(InputFormat::Mhs),
        bin_size: 10,
        mhs_pair: (0, 2),
        collect_het_sites: true,
        ..InputOptions::default()
    };
    let err = error(&two, &opts);
    assert!(err.contains("haplotypes"), "{err}");

    // Four haplotypes; the pair (0, 1) is heterozygous, homozygous, then
    // heterozygous on A.
    let four = dir.write(
        "four.mhs",
        "A\t20000\t19000\tACAC\nA\t20010\t10\tAACC\nA\t20030\t10\tCAAA\nB\t100\t50\tACAA\n",
    );
    let opts = InputOptions {
        mhs_pair: (0, 1),
        ..opts
    };
    let data = load(&four, &opts).unwrap();
    let SourceSummary::Mhs(m) = &data.source else {
        panic!("not an mhs summary")
    };
    assert_eq!((m.rows, m.het_rows, m.hom_rows, m.haplotypes), (4, 3, 1, 4));
    assert_eq!(m.uncalled_sites, 1000 + 10 + 50);
    assert_eq!(data.seqs.len(), 1);
    let a = &data.seqs[0];
    assert_eq!(
        (a.extent, a.called, a.het_sites),
        (Some(20_030), Some(19_020), Some(2))
    );
    assert_eq!(data.dropped[0].name, "B");
    // Same bins as the multihetsep reader for the kept sequence.
    assert_eq!(data.obs.rows[0], read_mhs(&four, None, 10).unwrap().rows[0]);
    // nr_called counts the called sites since the previous heterozygous row.
    let site = |pos, nr_called, alleles| HetSite {
        seq: 0,
        pos,
        nr_called,
        alleles,
    };
    assert_eq!(
        data.het_sites.unwrap(),
        [site(20_000, 19_000, *b"AC"), site(20_030, 20, *b"CA")]
    );
}

#[test]
fn mask_mode_het_sites_carry_alleles_in_gt_order() {
    let dir = TempDir::new("io_input");
    let vcf = dir.write(
        "in.vcf",
        format!(
            "{VCF_HEADER}\
             c\t3\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n\
             c\t5\t.\tA\tC\t50\tPASS\t.\tGT\t1|0\n\
             c\t7\t.\tA\tC,G\t50\tPASS\t.\tGT\t1/2\n\
             c\t9\t.\tA\tC\t50\tPASS\t.\tGT\t1/1\n\
             c\t10\t.\tA\tAT\t50\tPASS\t.\tGT\t0/1\n\
             c\t12\t.\tA\tC\t50\tPASS\t.\tGT\t1/0\n\
             d\t2\t.\tG\tT\t50\tPASS\t.\tGT\t0|1\n\
             e\t4\t.\tG\tT\t50\tPASS\t.\tGT\t0|1\n"
        ),
    );
    // Position 6 of c is not callable.
    let mask = dir.write("mask.bed", "c\t0\t5\nc\t6\t20\nd\t0\t1\ne\t0\t10\n");
    let opts = InputOptions {
        masks: vec![mask],
        collect_het_sites: true,
        filter: SeqFilter {
            min_good_bases: 2,
            min_good_fraction: 0.0,
        },
        ..vcf_opts(VcfMode::Mask)
    };
    let data = load(&vcf, &opts).unwrap();
    // d has one callable position and is dropped; e becomes sequence 1.
    let names: Vec<_> = data.seqs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["c", "e"]);
    let got: Vec<_> = data
        .het_sites
        .as_ref()
        .unwrap()
        .iter()
        .map(|h| (h.seq, h.pos, h.nr_called, &h.alleles))
        .collect();
    let want = [
        (0, 3, 3, b"AC"),
        (0, 5, 2, b"CA"),
        (0, 7, 1, b"CG"),
        // 8, 9 (hom-alt), 11, 12; the het indel at 10 is missing.
        (0, 12, 4, b"CA"),
        (1, 4, 4, b"GT"),
    ];
    assert_eq!(got, want);
    let c = &data.seqs[0];
    assert_eq!(
        (c.extent, c.called, c.het_sites),
        (Some(12), Some(10), Some(4))
    );
}

#[test]
fn het_sites_are_only_collected_on_request_and_not_for_psmcfa() {
    let dir = TempDir::new("io_input");
    let vcf = dir.write("in.vcf", call_c_vcf("c", 30, 7, 20));
    let opts = unfiltered(VcfMode::Auto);
    assert!(load(&vcf, &opts).unwrap().het_sites.is_none());
    let collect = InputOptions {
        collect_het_sites: true,
        ..opts
    };
    let sites = load(&vcf, &collect).unwrap().het_sites.unwrap();
    let pos: Vec<_> = sites.iter().map(|h| (h.pos, h.nr_called)).collect();
    assert_eq!(pos, [(7, 7), (14, 7), (21, 7), (28, 7)]);
    assert!(sites.iter().all(|h| h.alleles == *b"AG"));

    let psmcfa = dir.write("in.psmcfa", ">a\nTKT\n");
    let collect = InputOptions {
        collect_het_sites: true,
        ..InputOptions::default()
    };
    assert!(load(&psmcfa, &collect).unwrap().het_sites.is_none());
}

#[test]
fn psmcfa_input_is_not_filtered() {
    let dir = TempDir::new("io_input");
    let psmcfa = dir.write("in.psmcfa", ">a\nTKN\n>empty\n>b\nNNNNK\n");
    let data = load(&psmcfa, &InputOptions::default()).unwrap();
    let names: Vec<_> = data.seqs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!((data.seqs[1].n_bins, data.seqs[1].extent), (5, None));
    assert_eq!(data.dropped.len(), 1);
    assert_eq!(data.dropped[0].reason, DropReason::Empty);
    assert!(data.filter.is_none());
    let SourceSummary::Psmcfa(p) = &data.source else {
        panic!("not a psmcfa summary")
    };
    assert_eq!(
        (
            p.records,
            p.empty_records,
            p.bins,
            p.called_bins,
            p.het_bins
        ),
        (3, 1, 8, 3, 2)
    );
    assert_eq!(data.obs.rows, read_psmcfa(&psmcfa, None).unwrap().rows);
    data.ensure_usable().unwrap();
    assert_eq!((p.fastq_records, p.base_letters), (0, 0));
    assert!(data.warnings().is_empty(), "{:?}", data.warnings());
}

#[test]
fn a_per_base_consensus_given_as_psmcfa_is_pointed_out() {
    let dir = TempDir::new("io_input");
    let psmcfa_of = |text: &str| {
        let data = load(&dir.write("in.psmcfa", text), &InputOptions::default()).unwrap();
        let SourceSummary::Psmcfa(p) = data.source.clone() else {
            panic!("not a psmcfa summary")
        };
        (data, p)
    };
    // The FASTQ of vcfutils.pl vcf2fq: one character per base, lowercase
    // where filtered, with a quality block. It is read as C psmc reads it.
    let (data, p) = psmcfa_of("@chr1\nnnaaggTTCACRcga\n+\n!!!!!!IIIIII!!!\n");
    assert_eq!(
        data.obs.rows,
        [[2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
    );
    assert_eq!((p.fastq_records, p.base_letters), (1, 13));
    assert!(p.looks_like_per_base_consensus());
    let warnings = data.warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].contains("per-base consensus") && warnings[0].contains("fq2psmcfa first"),
        "{warnings:?}"
    );
    // A FASTA of bases, without quality.
    let (data, p) = psmcfa_of(">chr1\nACGTACGTTTGCAY\n");
    assert_eq!((p.fastq_records, p.base_letters), (0, 10));
    assert_eq!(data.warnings().len(), 1);
    // fq2psmcfa output, and stray letters in under 1% of the bins.
    let (data, p) = psmcfa_of(&format!(">chr1\nK{}A\n", "T".repeat(150)));
    assert_eq!(p.base_letters, 1);
    assert!(!p.looks_like_per_base_consensus());
    assert!(data.warnings().is_empty());
}

#[test]
fn batching_keeps_whole_sequences_available() {
    let dir = TempDir::new("io_input");
    let psmcfa = dir.write("in.psmcfa", ">a\nTKNTKNT\n>b\nKK\n");
    let batched = |batch_size| {
        let opts = InputOptions {
            batch_size: Some(batch_size),
            ..InputOptions::default()
        };
        load(&psmcfa, &opts)
    };
    let data = batched(3).unwrap();
    assert_eq!(data.obs.rows.len(), 4);
    assert_eq!(data.obs.row_starts, [true, false, false, true]);
    let seqs = data.sequences();
    assert_eq!(seqs[0].1.as_ref(), &[0, 1, 2, 0, 1, 2, 0]);
    assert_eq!(seqs[1].1.as_ref(), &[1, 1]);
    assert_eq!((data.n_bins(), data.het_bins()), (9, 4));
    let err = batched(0).unwrap_err();
    assert!(format!("{err:#}").contains("batch_size"), "{err:#}");
}

#[test]
fn input_without_heterozygous_bins_loads_but_is_not_usable() {
    let dir = TempDir::new("io_input");
    let psmcfa = dir.write("in.psmcfa", ">a\nTTNT\n");
    let data = load(&psmcfa, &InputOptions::default()).unwrap();
    assert!(
        data.warnings()
            .iter()
            .any(|w| w.contains("no bin is heterozygous"))
    );
    let msg = format!("{:#}", data.ensure_usable().unwrap_err());
    assert!(msg.contains("heterozygous"), "{msg}");

    // Mask mode keeps the reader's detailed message.
    let vcf = dir.write(
        "in.vcf",
        format!("{VCF_HEADER}c\t5\t.\tA\tC\t50\tq10\t.\tGT\t0/1\n"),
    );
    let data = load(&vcf, &unfiltered(VcfMode::Mask)).unwrap();
    let msg = format!("{:#}", data.ensure_usable().unwrap_err());
    assert!(msg.contains("FILTER other than PASS"), "{msg}");
}

/// `call_c_vcf` with `name` (raw bytes) as the chromosome.
fn vcf_bytes_with_chrom(name: &[u8]) -> Vec<u8> {
    let text = call_c_vcf("@", 20, 5, 20);
    let mut out = Vec::new();
    for line in text.split_inclusive('\n') {
        match line.strip_prefix('@') {
            Some(rest) => {
                out.extend_from_slice(name);
                out.extend_from_slice(rest.as_bytes());
            }
            None => out.extend_from_slice(line.as_bytes()),
        }
    }
    out
}

#[test]
fn chromosome_names_must_be_utf8() {
    // Latin-1 'chré': a lossy conversion would make it 'chr\u{FFFD}', the
    // name of every other such chromosome too.
    let dir = TempDir::new("io_input");
    let latin1 = dir.write("latin1.vcf", vcf_bytes_with_chrom(b"chr\xe9"));
    for mode in [VcfMode::Auto, VcfMode::Consensus, VcfMode::Mask] {
        let msg = error(&latin1, &vcf_opts(mode));
        assert!(
            msg.contains("chromosome name 'chr\\xe9' is not valid UTF-8"),
            "{mode:?}: {msg}"
        );
    }
    assert!(read_vcf(&latin1, None, 10, None, &[], &[]).is_err());

    // UTF-8 names are kept byte for byte, as vcf2fq and fq2psmcfa keep them.
    let utf8 = dir.write("utf8.vcf", vcf_bytes_with_chrom("chré".as_bytes()));
    for mode in [VcfMode::Consensus, VcfMode::Mask] {
        let opts = InputOptions {
            bin_size: 5,
            ..unfiltered(mode)
        };
        let data = load(&utf8, &opts).unwrap();
        assert_eq!(data.seqs[0].name, "chré");
        let mut out = Vec::new();
        write_psmcfa_to(
            &mut out,
            [(data.seqs[0].name.as_str(), data.obs.rows[0].as_slice())],
        )
        .unwrap();
        assert_eq!(out, b">chr\xc3\xa9\nKKKK\n");
    }
}
