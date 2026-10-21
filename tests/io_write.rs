//! The psmcfa and multihetsep writers: exact layout, gzip by file name, and
//! errors. Byte equality with fq2psmcfa and generate_multihetsep.py on real
//! data is in `tests/io_route_a.rs`.

mod common;

use common::{TempDir, gunzipped};
use psmc_rs::io::input::{Dataset, InputFormat, InputOptions, SeqFilter, load};
use psmc_rs::io::psmcfa::read_psmcfa_records;
use psmc_rs::io::write::{
    MultihetsepOptions, write_multihetsep, write_multihetsep_to, write_psmcfa, write_psmcfa_to,
};
use std::fs;

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

/// `text` loaded from a file `name` with `opts`.
fn load_text(name: &str, text: &str, opts: &InputOptions) -> Dataset {
    let dir = TempDir::new("io_write");
    load(&dir.write(name, text), opts).unwrap()
}

/// Multihetsep rows of the sequence `name` only.
fn only(name: &str) -> MultihetsepOptions {
    MultihetsepOptions {
        sequence: Some(name.to_string()),
    }
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
    for (name, bins, error) in [
        ("a", &[0u8, 3][..], "invalid bin symbol 3"),
        ("chr 1", &[0], "whitespace"),
    ] {
        let err = write_psmcfa_to(&mut Vec::new(), [(name, bins)]).unwrap_err();
        assert!(format!("{err:#}").contains(error), "{err:#}");
    }
}

#[test]
fn files_ending_in_gz_are_gzip() {
    let data = load_text(
        "in.psmcfa",
        ">a\nTTKN\n>b\nKKKK\n",
        &InputOptions::default(),
    );
    let dir = TempDir::new("io_write");
    let (plain, gz) = (dir.path("out.psmcfa"), dir.path("out.psmcfa.gz"));
    write_psmcfa(&plain, &data).unwrap();
    write_psmcfa(&gz, &data).unwrap();
    let text = fs::read(&plain).unwrap();
    assert_eq!(text, b">a\nTTKN\n>b\nKKKK\n");
    // `gunzipped` checks the gzip magic of a `.gz` path.
    assert_eq!(gunzipped(&gz), text);
    assert_eq!(read_psmcfa_records(&gz).unwrap().len(), 2);
}

#[test]
fn batched_sequences_are_written_whole() {
    let long: String = "TKN".repeat(50);
    let opts = InputOptions {
        batch_size: Some(7),
        ..InputOptions::default()
    };
    let data = load_text("in.psmcfa", &format!(">a\n{long}\n>b\nTT\n"), &opts);
    assert!(data.obs.rows.len() > 2);
    let dir = TempDir::new("io_write");
    let out = dir.path("out.psmcfa");
    write_psmcfa(&out, &data).unwrap();
    assert_eq!(
        fs::read_to_string(&out).unwrap(),
        fq2psmcfa_layout("a", &long) + &fq2psmcfa_layout("b", "TT")
    );
}

fn mhs_dataset() -> Dataset {
    let opts = InputOptions {
        format: Some(InputFormat::Vcf),
        collect_het_sites: true,
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    load_text(
        "in.vcf",
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
         c1\t10\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n\
         c1\t25\t.\tG\tT\t50\tPASS\t.\tGT\t1|0\n\
         c2\t7\t.\tA\tG\t50\tPASS\t.\tGT\t1/1\n\
         c2\t8\t.\tC\tT\t50\tPASS\t.\tGT\t0/1\n",
        &opts,
    )
}

/// A psmcfa dataset, which has no positions to write as multihetsep.
fn psmcfa_dataset(collect_het_sites: bool) -> Dataset {
    let opts = InputOptions {
        collect_het_sites,
        ..InputOptions::default()
    };
    load_text("in.psmcfa", ">a\nTK\n", &opts)
}

#[test]
fn multihetsep_layout_is_generate_multihetsep_s() {
    let data = mhs_dataset();
    let text = |opts: &MultihetsepOptions| {
        let mut out = Vec::new();
        write_multihetsep_to(&mut out, &data, opts).unwrap();
        String::from_utf8(out).unwrap()
    };
    assert_eq!(
        text(&MultihetsepOptions::default()),
        "c1\t10\t10\tAC\nc1\t25\t15\tTG\nc2\t8\t8\tCT\n"
    );
    assert_eq!(text(&only("c2")), "c2\t8\t8\tCT\n");

    let dir = TempDir::new("io_write");
    let gz = dir.path("out.multihetsep.gz");
    write_multihetsep(&gz, &data, &only("c2")).unwrap();
    assert_eq!(gunzipped(&gz), b"c2\t8\t8\tCT\n");
}

#[test]
fn multihetsep_writer_errors() {
    let err = write_multihetsep_to(&mut Vec::new(), &mhs_dataset(), &only("c3")).unwrap_err();
    assert!(
        format!("{err:#}").contains("no kept sequence is named 'c3'"),
        "{err:#}"
    );
    let data = psmcfa_dataset(true);
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
    let without_sites = psmcfa_dataset(false);
    let mut bad_name = data.clone();
    bad_name.seqs[0].name = "chr 1".to_string();
    let mut bad_bin = data.clone();
    bad_bin.obs.rows[0][0] = 3;

    let dir = TempDir::new("io_write");
    for ext in ["out", "out.gz"] {
        let existing = dir.path(&format!("existing.{ext}"));
        let content = b"output of an earlier run\n";
        fs::write(&existing, content).unwrap();
        let all = MultihetsepOptions::default();
        for (err, text) in [
            (
                write_multihetsep(&existing, &data, &only("chrX")),
                "no kept sequence is named 'chrX'",
            ),
            (
                write_multihetsep(&existing, &without_sites, &all),
                "loaded without heterozygous sites",
            ),
            (write_psmcfa(&existing, &bad_name), "whitespace"),
            (write_psmcfa(&existing, &bad_bin), "invalid bin symbol 3"),
        ] {
            let msg = format!("{:#}", err.unwrap_err());
            assert!(
                msg.contains(text) && msg.contains("failed to write"),
                "{msg}"
            );
        }
        assert_eq!(fs::read(&existing).unwrap(), content, "{ext}");

        // A valid request still replaces the file.
        write_multihetsep(&existing, &data, &all).unwrap();
        assert_ne!(fs::read(&existing).unwrap(), content, "{ext}");
    }
}
