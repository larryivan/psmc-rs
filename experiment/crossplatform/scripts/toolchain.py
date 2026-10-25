"""Write toolchain.txt: the laptop, the two psmc-rs builds that ran, and the laptop C psmc binary's forensics.

Usage: toolchain.py [--built-at COMMIT]
           after the builds, with bin/psmc-rs from the default toolchain and bin/psmc-rs-rustc1.99
           from rustc 1.99.0, both `cargo build --release --locked`. "git HEAD" is the commit the
           builds were made at: HEAD, or COMMIT when run again later (its src, Cargo.toml and
           Cargo.lock must be HEAD's).
       toolchain.py --digest FILE...
           prints each binary's digest without LC_UUID and code signature (below), to check a rebuild.
Paths are written as <repo> (this repository) and <xplat> (this directory).

A binary's whole-file sha256 depends on the target directory it was built in: rebuilt from the
same sources by the same toolchain in another target directory, it differed in the Mach-O UUID
(LC_UUID), and so in the ad-hoc code signature, and nowhere else. The digest "without LC_UUID
and code signature" is the sha256 of the file with those bytes zeroed, which that rebuild
reproduced. Both digests also depend on the source paths the binary holds in its panic
messages, under the builder's home directory (~/.cargo, ~/.rustup): another user's or
machine's build is not expected to reproduce either.
"""
import hashlib
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # the repository root
X = Path(__file__).resolve().parent.parent
C_TOOLS = REPO / "experiment/.tools/psmc_c"  # the laptop's C psmc builds (not in the repository)
BUILDS = (("bin/psmc-rs", None), ("bin/psmc-rs-rustc1.99", "+1.99"))


def run(*cmd, cwd=REPO):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd).stdout.strip()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(p: Path) -> str:
    """sha256 of a 64-bit Mach-O binary with its LC_UUID and its code signature zeroed."""
    b = bytearray(p.read_bytes())
    magic, ncmds = struct.unpack_from("<I", b, 0)[0], struct.unpack_from("<I", b, 16)[0]
    assert magic == 0xFEEDFACF, f"{p}: not a 64-bit Mach-O file"
    off, seen = 32, set()
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<2I", b, off)
        if cmd == 0x1B:  # LC_UUID: 16 bytes after cmd and cmdsize
            b[off + 8:off + 24] = bytes(16)
            seen.add("uuid")
        elif cmd == 0x1D:  # LC_CODE_SIGNATURE: dataoff, datasize
            dataoff, datasize = struct.unpack_from("<2I", b, off + 8)
            b[dataoff:dataoff + datasize] = bytes(datasize)
            seen.add("signature")
        off += size
    assert seen == {"uuid", "signature"}, (p, seen)
    return hashlib.sha256(b).hexdigest()


if sys.argv[1:2] == ["--digest"]:
    for f in sys.argv[2:]:
        print(f"{digest(Path(f))}  {f}")
    sys.exit(0)
built_at = sys.argv[sys.argv.index("--built-at") + 1] if "--built-at" in sys.argv else "HEAD"
TREES = ("src", "Cargo.toml", "Cargo.lock")
assert all(run("git", "rev-parse", f"{built_at}:{t}") == run("git", "rev-parse", f"HEAD:{t}") for t in TREES), \
    f"{built_at}'s sources are not HEAD's"

out = []
w = out.append
w("== laptop")
w(run("sysctl", "-n", "machdep.cpu.brand_string"))
w(f"cores: {run('sysctl', '-n', 'hw.ncpu')} (P {run('sysctl', '-n', 'hw.perflevel0.physicalcpu')}, "
  f"E {run('sysctl', '-n', 'hw.perflevel1.physicalcpu')}); mem {int(run('sysctl', '-n', 'hw.memsize')) >> 30} GB")
w(run("sw_vers"))
w(run("uname", "-srvm"))
w("")
w(f"git HEAD {run('git', 'rev-parse', built_at)}")
ids = " ".join(f"{p}={run('git', 'rev-parse', 'HEAD:' + p)}" for p in TREES)
dirty = run("git", "status", "--porcelain", "--", *TREES)
w(f"{ids} dirty_files={len(dirty.splitlines())}")
w(" ".join(f"{k}={os.environ.get(k, '')!r}" for k in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS")))
cfg = REPO / ".cargo/config.toml"
w(".cargo/config.toml: " + (f"{sum(1 for l in cfg.read_text().splitlines() if l.strip() and not l.strip().startswith('#'))} non-comment lines"
                            if cfg.exists() else "none"))
for path, tc in BUILDS:
    tcargs = [tc] if tc else []
    b = X / path
    w("")
    w(f"== build <xplat>/{path}: cargo {tc + ' ' if tc else ''}build --release --locked")
    w(run("rustc", *tcargs, "-vV"))
    w(run("cargo", *tcargs, "--version"))
    w(f"{sha(b)}  <xplat>/{path}")
    w(f"sha256 without LC_UUID and code signature: {digest(b)}  <xplat>/{path}")
    w(run(str(b), "--version"))
    feats = re.findall(r'target_feature="[^"]+"', run("rustc", *tcargs, "--print", "cfg"))
    w("rustc --print cfg target_feature: " + " ".join(feats))
w("")
w("== laptop C psmc (frozen .production run)")
w(f"{sha(C_TOOLS / 'psmc')}  experiment/.tools/psmc_c/psmc")
for name in ("psmc", "psmc_O", "psmc_O1", "psmc_O0"):
    # the hex dump of the __text section (otool -tX: no header line)
    text = subprocess.run(["otool", "-tX", str(C_TOOLS / name)], capture_output=True).stdout
    w(f"{name} __text sha256: {hashlib.sha256(text).hexdigest()[:16]}")
dis = run("otool", "-tv", str(C_TOOLS / "psmc"))
fma = re.findall(r"\tf(?:n?)m(?:add|sub)\t", dis)
w(f"fused multiply-add instructions in psmc: {len(fma)}")
prod = re.search(r'DW_AT_producer\s+\("([^"]+)"\)', run("dwarfdump", "--debug-info", str(C_TOOLS / "psmc_asan.dSYM")))
w(f"sibling ASan build DWARF producer: {prod.group(1) if prod else '?'}")
w(f"current clang: {run('clang', '--version').splitlines()[0]}")
text = "\n".join(out) + "\n"
(X / "toolchain.txt").write_text(text.replace(str(X), "<xplat>").replace(str(REPO), "<repo>"))
print(text)
