## Summary

<!-- What does this change and why? Link the issue it addresses (e.g. "Fixes #12"). -->

## Changes

<!-- One bullet per logical change. -->

-

## Testing

<!-- Which tests cover the change? For a behavioural fix, confirm that the new
     test fails without the fix and passes with it. For performance changes,
     give old vs new timings (median of >= 3 interleaved runs) and the machine. -->

## Checklist

- [ ] `cargo fmt --all` leaves no diff
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo test --workspace` passes, including `tests/c_equivalence.rs` (tolerances and reference outputs unchanged)
- [ ] Behavioural fixes come with a test that fails before and passes after
- [ ] New public items have rustdoc; user-visible changes are listed in `CHANGELOG.md` under *Unreleased*
- [ ] If `experiment/` changed: `python3 -m unittest discover -s experiment/tests` passes
