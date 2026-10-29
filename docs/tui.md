# The terminal interface

```bash
psmc-rs --tui [INPUT] [OUTPUT] [N_ITER] [OPTIONS]
```

opens a terminal interface for the same fit. Its three pages are Setup
(F1), Run (F2) and Result (F3), with a line of the keys that apply.

**Setup.** The form takes the input, the output, N_ITER (25), the pattern
(empty for `4+5*3+4`), the generation time, the threads, the input format,
the bin size, μ and the smoothing. Arguments and options given with `--tui`
fill it, and any other option is kept; the page shows the equivalent
command line, or what the form still lacks. Every character types into the
selected field:

| key | action |
| --- | --- |
| Tab, Shift+Tab, Up, Down | move between fields |
| Ctrl+U | clear a field |
| Space, Left, Right | change the input format |
| Ctrl+O, Ctrl+D | pick the input file, the output directory |
| F5, Enter | run |
| Esc | quit |

**Run.** x or Ctrl+C stops a run: EM and the bootstrap stop at their next
check, and the run shows "Cancelled". Ctrl+C with no run quits. A run that
fails goes back to the Setup page with the error in its log; one that
finishes opens the Result page, which lists what it wrote.

On the Run and Result pages, q quits and r runs; Up/Down and
PageUp/PageDown scroll the log, End goes back to its end, and Ctrl+L
clears it.

`--qc-json -` is refused with `--tui`, whose screen is stdout.
