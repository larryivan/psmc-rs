# psmc-rs documentation

Start with the [README](../README.md): what psmc-rs does, how to install it,
a quick start and how to read the results. These pages go further.

| page | what it covers |
| --- | --- |
| [preprocessing.md](preprocessing.md) | from aligned reads to an input, by two routes, and the pitfalls (coverage, depth, masks, sex chromosomes, repeats, μ and the generation time) |
| [inputs.md](inputs.md) | how psmcfa, VCF and multihetsep are read; masks; missing data and partly callable bins; `psmc-rs convert` |
| [fitting.md](fitting.md) | the model options, the starting point, the M-step, the bootstrap |
| [outputs.md](outputs.md) | every output file (JSON, HTML report, `.psmc`, TMRCA, bootstrap files), the console and the exit status |
| [qc.md](qc.md) | `psmc-rs qc`: the report, and what to do about each finding |
| [tui.md](tui.md) | the terminal interface |
| [performance.md](performance.md) | threads, memory, determinism and the CPU kernels |
| [validation.md](validation.md) | agreement with C psmc and the classic tools, the tests, and what the manuscript's experiments measured |
| [options.md](options.md) | every option, in brief |
| [design/](design/README.md) | design notes for developers |
