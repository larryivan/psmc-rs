//! Reading C psmc's `.psmc` text output back, for the tests that check it.

/// The tag of a line (`RD`, `LK`, `RS`, ...).
pub fn tag(line: &str) -> &str {
    line.split('\t').next().unwrap()
}

/// The lines of round `round`, from `RD` to `//`.
pub fn round_lines(text: &str, round: usize) -> Vec<&str> {
    let start = format!("RD\t{round}");
    let mut lines: Vec<&str> = text
        .lines()
        .skip_while(|l| *l != start)
        .take_while(|l| *l != "//")
        .collect();
    assert!(!lines.is_empty(), "no round {round}");
    lines.push("//");
    lines
}

/// The numbers of one round.
#[derive(Debug)]
pub struct Round {
    pub lk: f64,
    pub ri: f64,
    pub tr: (f64, f64),
    pub mt: f64,
    /// RS rows: t_k, lambda_k, n_recomb sigma_k, sigma_k, post_sigma_k.
    pub rs: Vec<[f64; 5]>,
}

pub fn parse_round(lines: &[&str]) -> Round {
    let field = |t: &str| {
        lines
            .iter()
            .find_map(|l| l.strip_prefix(&format!("{t}\t")))
            .unwrap_or_else(|| panic!("no {t} line"))
    };
    let num = |s: &str| s.parse::<f64>().unwrap();
    let (theta, rho) = field("TR").split_once('\t').unwrap();
    let rs = lines
        .iter()
        .filter_map(|l| l.strip_prefix("RS\t"))
        .enumerate()
        .map(|(k, l)| {
            let cols: Vec<&str> = l.split('\t').collect();
            assert_eq!(cols[0], k.to_string());
            [1, 2, 3, 4, 5].map(|i| num(cols[i]))
        })
        .collect();
    Round {
        lk: num(field("LK")),
        ri: num(field("RI")),
        tr: (num(theta), num(rho)),
        mt: num(field("MT")),
        rs,
    }
}

/// Every round of a `.psmc` file, in order.
pub fn parse_rounds(text: &str) -> Vec<Round> {
    let n = text.lines().filter(|l| tag(l) == "RD").count();
    (0..n).map(|i| parse_round(&round_lines(text, i))).collect()
}

/// The value of `key:` in the header's `MM` lines.
pub fn header_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("{key}:");
    text.lines()
        .take_while(|l| tag(l) != "RD")
        .filter(|l| tag(l) == "MM")
        .find_map(|l| {
            let at = l.find(&needle)?;
            let rest = &l[at + needle.len()..];
            Some(rest.split(',').next().unwrap().trim())
        })
}
