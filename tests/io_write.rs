//! The psmcfa and multihetsep writers: exact layout, gzip by file name, and
//! errors. Byte equality with fq2psmcfa and generate_multihetsep.py on real
//! data is in `tests/io_route_a.rs`.

use psmc_rs::io::input::{InputFormat, InputOptions, SeqFilter, load};
use psmc_rs::io::psmcfa::read_psmcfa_records;
use psmc_rs::io::write::{
    MultihetsepOptions, write_multihetsep, write_multihetsep_to, write_psmcfa, write_psmcfa_to,
};
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// A path no other test of this process uses (tests run in parallel, and the
/// clock alone is too coarse to tell them apart).
fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()))
}

fn psmcfa_text(seqs: &[(&str, &[u8])]) -> String {
    let mut out = Vec::new();
    write_psmcfa_to(&mut out, seqs.iter().copied()).unwrap();
    String::from_utf8(out).unwrap()
}

/// fq2psmcfa's loop: `>name`, a newline before every symbol whose index is
/// a multiple of 60, a final newline.
fn fq2psmcfa_layout(name: &str, symbols: &str) -> String {
    let mut out = format!(">{name}");
    for (i, c) in symbols.chars().enumerate() {
        if i % 60 == 0 {
            out.push('\n');
        }
        out.push(c);
    }
    out.push('\n');
    out
}

#[test]
fn psmcfa_layout_is_fq2psmcfa_s() {
    for n in [1usize, 2, 59, 60, 61, 119, 120, 121, 1000] {
        let bins: Vec<u8> = (0..n).map(|i| [0, 1, 2][i % 7 % 3]).collect();
        let symbols: String = bins.iter().map(|&b| ['T', 'K', 'N'][b as usize]).collect();
        assert_eq!(
            psmcfa_text(&[("chr1", &bins)]),
            fq2psmcfa_layout("chr1", &symbols),
            "{n} bins"
        );
    }
    assert_eq!(
        psmcfa_text(&[("a", &[0, 1]), ("b", &[2])]),
        ">a\nTK\n>b\nN\n"
    );
    // No bins: the header and the final newline.
    assert_eq!(psmcfa_text(&[("e", &[])]), ">e\n");
}

#[test]
fn psmcfa_writer_rejects_what_it_cannot_write() {
    let mut out = Vec::new();
    let err = write_psmcfa_to(&mut out, [("a", &[0u8, 3][..])]).unwrap_err();
    assert!(
        format!("{err:#}").contains("invalid bin symbol 3"),
        "{err:#}"
    );
    let err = write_psmcfa_to(&mut Vec::new(), [("chr 1", &[0u8][..])]).unwrap_err();
    assert!(format!("{err:#}").contains("whitespace"), "{err:#}");
}

#[test]
fn files_ending_in_gz_are_gzip() {
    let input = unique_temp_path("write_in", "psmcfa");
    fs::write(&input, ">a\nTTKN\n>b\nKKKK\n").unwrap();
    let data = load(&input, &InputOptions::default()).unwrap();
    let _ = fs::remove_file(&input);

    let plain = unique_temp_path("write_plain", "psmcfa");
    let gz = unique_temp_path("write_gz", "psmcfa.gz");
    write_psmcfa(&plain, &data).unwrap();
    write_psmcfa(&gz, &data).unwrap();
    let text = fs::read(&plain).unwrap();
    assert_eq!(text, b">a\nTTKN\n>b\nKKKK\n");
    let raw = fs::read(&gz).unwrap();
    assert_eq!(&raw[..2], &[0x1f, 0x8b]);
    let mut unzipped = Vec::new();
    flate2::read::MultiGzDecoder::new(raw.as_slice())
        .read_to_end(&mut unzipped)
        .unwrap();
    assert_eq!(unzipped, text);
    let records = read_psmcfa_records(&gz).unwrap();
    assert_eq!(records.len(), 2);
    let _ = fs::remove_file(plain);
    let _ = fs::remove_file(gz);
}

#[test]
fn batched_sequences_are_written_whole() {
    let input = unique_temp_path("write_batch", "psmcfa");
    let long: String = "TKN".repeat(50);
    fs::write(&input, format!(">a\n{long}\n>b\nTT\n")).unwrap();
    let opts = InputOptions {
        batch_size: Some(7),
        ..InputOptions::default()
    };
    let data = load(&input, &opts).unwrap();
    let _ = fs::remove_file(&input);
    assert!(data.obs.rows.len() > 2);
    let out = unique_temp_path("write_batch", "psmcfa");
    write_psmcfa(&out, &data).unwrap();
    let text = fs::read_to_string(&out).unwrap();
    let _ = fs::remove_file(out);
    assert_eq!(
        text,
        format!(
            "{}{}",
            fq2psmcfa_layout("a", &long),
            fq2psmcfa_layout("b", "TT")
        )
    );
}

fn mhs_dataset() -> psmc_rs::io::input::Dataset {
    let vcf = unique_temp_path("write_mhs", "vcf");
    fs::write(
        &vcf,
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
         c1\t10\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n\
         c1\t25\t.\tG\tT\t50\tPASS\t.\tGT\t1|0\n\
         c2\t7\t.\tA\tG\t50\tPASS\t.\tGT\t1/1\n\
         c2\t8\t.\tC\tT\t50\tPASS\t.\tGT\t0/1\n",
    )
    .unwrap();
    let opts = InputOptions {
        format: Some(InputFormat::Vcf),
        collect_het_sites: true,
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let data = load(&vcf, &opts).unwrap();
    let _ = fs::remove_file(vcf);
    data
}

#[test]
fn multihetsep_layout_is_generate_multihetsep_s() {
    let data = mhs_dataset();
    let mut out = Vec::new();
    write_multihetsep_to(&mut out, &data, &MultihetsepOptions::default()).unwrap();
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "c1\t10\t10\tAC\nc1\t25\t15\tTG\nc2\t8\t8\tCT\n"
    );
    let mut out = Vec::new();
    let one = MultihetsepOptions {
        sequence: Some("c2".to_string()),
    };
    write_multihetsep_to(&mut out, &data, &one).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "c2\t8\t8\tCT\n");

    let gz = unique_temp_path("write_mhs", "multihetsep.gz");
    write_multihetsep(&gz, &data, &one).unwrap();
    let raw = fs::read(&gz).unwrap();
    let _ = fs::remove_file(gz);
    let mut text = String::new();
    flate2::read::MultiGzDecoder::new(raw.as_slice())
        .read_to_string(&mut text)
        .unwrap();
    assert_eq!(text, "c2\t8\t8\tCT\n");
}

#[test]
fn multihetsep_writer_errors() {
    let data = mhs_dataset();
    let missing = MultihetsepOptions {
        sequence: Some("c3".to_string()),
    };
    let err = write_multihetsep_to(&mut Vec::new(), &data, &missing).unwrap_err();
    assert!(format!("{err:#}").contains("no kept sequence is named 'c3'"));

    let input = unique_temp_path("write_mhs_err", "psmcfa");
    fs::write(&input, ">a\nTK\n").unwrap();
    let opts = InputOptions {
        collect_het_sites: true,
        ..InputOptions::default()
    };
    let data = load(&input, &opts).unwrap();
    let _ = fs::remove_file(input);
    let err =
        write_multihetsep_to(&mut Vec::new(), &data, &MultihetsepOptions::default()).unwrap_err();
    assert!(
        format!("{err:#}").contains("psmcfa input has no positions"),
        "{err:#}"
    );
}

#[test]
fn rejected_writes_leave_an_existing_file_as_it_was() {
    let data = mhs_dataset();
    let without_sites = {
        let input = unique_temp_path("write_keep", "psmcfa");
        fs::write(&input, ">a\nTK\n").unwrap();
        let data = load(&input, &InputOptions::default()).unwrap();
        let _ = fs::remove_file(input);
        data
    };
    let mut bad_name = data.clone();
    bad_name.seqs[0].name = "chr 1".to_string();
    let mut bad_bin = data.clone();
    bad_bin.obs.rows[0][0] = 3;

    for ext in ["out", "out.gz"] {
        let existing = unique_temp_path("write_keep", ext);
        let content = b"output of an earlier run\n";
        fs::write(&existing, content).unwrap();
        let missing = MultihetsepOptions {
            sequence: Some("chrX".to_string()),
        };
        let errors = [
            write_multihetsep(&existing, &data, &missing).unwrap_err(),
            write_multihetsep(&existing, &without_sites, &MultihetsepOptions::default())
                .unwrap_err(),
            write_psmcfa(&existing, &bad_name).unwrap_err(),
            write_psmcfa(&existing, &bad_bin).unwrap_err(),
        ];
        let expected = [
            "no kept sequence is named 'chrX'",
            "loaded without heterozygous sites",
            "whitespace",
            "invalid bin symbol 3",
        ];
        for (err, text) in errors.iter().zip(expected) {
            let msg = format!("{err:#}");
            assert!(
                msg.contains(text) && msg.contains("failed to write"),
                "{msg}"
            );
        }
        assert_eq!(fs::read(&existing).unwrap(), content, "{ext}");

        // A valid request still replaces the file.
        write_multihetsep(&existing, &data, &MultihetsepOptions::default()).unwrap();
        assert_ne!(fs::read(&existing).unwrap(), content, "{ext}");
        let _ = fs::remove_file(existing);
    }
}
