"""Tests for experiment.pipeline.tools: identities and exact command lines."""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, tools  # noqa: E402

FAKE_C = tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5"})
FAKE_RS = tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0"})


def in_git() -> bool:
    if shutil.which("git") is None:
        return False
    return subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT, capture_output=True).returncode == 0


class Commands(unittest.TestCase):
    def test_c_psmc(self):
        cmd = tools.c_psmc_fit(FAKE_C, config.INFERENCE, Path("in.psmcfa"), Path("out.psmc"))
        self.assertEqual(cmd, ["/x/psmc", "-N25", "-t15", "-r5", "-p", "4+25*2+4+6", "-o", "out.psmc", "in.psmcfa"])

    def test_psmc_rs(self):
        cmd = tools.rs02_fit(FAKE_RS, config.INFERENCE, config.SIMULATION_SCALING, Path("in"), Path("o.json"),
                             threads=3, psmc_out=Path("o.psmc"), extra=["--qc-json", "q.json"])
        self.assertEqual(cmd[:4], ["/x/psmc-rs", "in", "o.json", "25"])
        opts = dict(zip(cmd[4::2], cmd[5::2]))
        self.assertEqual(opts["--pattern"], "4+25*2+4+6")
        self.assertEqual(opts["--t-max"], "15")
        self.assertEqual(opts["--tr-ratio"], "5")
        self.assertEqual(opts["--smooth-lambda"], "0")
        self.assertEqual(opts["--mstep-iters"], "1000")  # psmc-rs's default is 100; see config.Inference
        self.assertEqual(opts["--bin-size"], "100")
        self.assertEqual(opts["--mu"], "2.5e-08")
        self.assertEqual(opts["--gen-years"], "25")
        self.assertEqual(opts["--threads"], "3")
        self.assertIn("--no-progress", cmd)
        self.assertEqual(cmd[-4:], ["--psmc-out", "o.psmc", "--qc-json", "q.json"])

    def test_qc(self):
        self.assertEqual(tools.rs02_qc(FAKE_RS, Path("a.vcf.gz"), Path("q.json")), ["/x/psmc-rs", "qc", "a.vcf.gz", "--json", "q.json"])

    def test_fmt(self):
        self.assertEqual([tools.fmt(x) for x in (15.0, 5.0, 2.5e-8, 0.0, 1e-3)], ["15", "5", "2.5e-08", "0", "0.001"])


@unittest.skipUnless(in_git(), "not a git checkout")
class Identities(unittest.TestCase):
    def test_rs02_source_state(self):
        state = tools.rs02_source_state()
        self.assertRegex(state["source_tree"], r"^src=[0-9a-f]{40};Cargo.toml=[0-9a-f]{40};Cargo.lock=[0-9a-f]{40}$")
        self.assertTrue(state["source_dirty"] == "" or re.match(r"^[0-9a-f]{64}$", state["source_dirty"]))

    def test_rs02_identity_is_its_sources_and_toolchain_not_a_commit_or_a_path(self):
        binary = ROOT / "target" / "release" / "psmc-rs"
        if not binary.exists():
            self.skipTest("target/release/psmc-rs not built")
        t = tools.psmc_rs_02(build=False)
        self.assertEqual(set(t.identity), {"version", "source_tree", "source_dirty", *tools.BUILD_FIELDS})
        self.assertRegex(t.info["source_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(t.info["binary_sha256"], tools.sha256_file(binary))
        with mock.patch.dict("os.environ", {"RUSTFLAGS": "-C opt-level=2"}):
            self.assertNotEqual(tools.psmc_rs_02(build=False).identity, t.identity)

    def test_psmc_rs_outside_a_checkout_is_identified_by_its_binary(self):
        binary = ROOT / "target" / "release" / "psmc-rs"
        if not binary.exists():
            self.skipTest("target/release/psmc-rs not built")
        with mock.patch.object(tools, "rs02_source_state", side_effect=tools.ToolError("not a repository")):
            t = tools.psmc_rs_02(build=False)
        self.assertEqual(t.identity["binary_sha256"], tools.sha256_file(binary))
        self.assertIn("unknown", t.identity["source_tree"])

    def test_missing_tools_are_reported(self):
        with self.assertRaises(tools.ToolError):
            tools.c_psmc(Path("/nonexistent/tools"))
        with self.assertRaises(tools.ToolError):
            tools.utility("fq2psmcfa", Path("/nonexistent/tools"))


def fake_program(path: Path, output: str) -> Path:
    """An executable that prints ``output`` to stdout and exits 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho '{output}'\n")
    path.chmod(0o755)
    return path


# Variables that would leak the test process's own build settings into toolchain().
BUILD_VARS = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS")


def without_build_vars() -> dict:
    import os

    return {k: v for k, v in os.environ.items()
            if k not in BUILD_VARS and not k.startswith(("CARGO_PROFILE_RELEASE_", "CARGO_TARGET_"))}


class Toolchain(unittest.TestCase):
    """What a psmc-rs build depends on beyond its sources: rustc, the flags cargo passes, the target features."""

    def test_effective_rustflags_follow_cargos_precedence(self):
        host = "x86_64-unknown-linux-gnu"
        self.assertEqual(tools.target_rustflags_var(host), "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS")
        env = {"CARGO_BUILD_RUSTFLAGS": "-C target-cpu=znver2"}
        self.assertEqual(tools.effective_rustflags(env, host), ["-C", "target-cpu=znver2"])
        env[tools.target_rustflags_var(host)] = "-C target-feature=+fma"
        self.assertEqual(tools.effective_rustflags(env, host), ["-C", "target-feature=+fma"])
        env["RUSTFLAGS"] = ""  # set, even empty: cargo ignores the others
        self.assertEqual(tools.effective_rustflags(env, host), [])
        env["CARGO_ENCODED_RUSTFLAGS"] = "-C\x1ftarget-cpu=native"
        self.assertEqual(tools.effective_rustflags(env, host), ["-C", "target-cpu=native"])
        self.assertEqual(tools.effective_rustflags({}, host), [])

    def test_target_features_are_read_from_rustc(self):
        cfg = 'debug_assertions\ntarget_arch="x86_64"\ntarget_feature="avx"\ntarget_feature="avx2"\ntarget_feature="fma"\n'
        done = subprocess.CompletedProcess([], 0, stdout=cfg, stderr="")
        with mock.patch.object(tools, "_run", return_value=done) as run:
            self.assertEqual(tools.target_features(["-C", "target-cpu=native"]), "avx2,fma")
        self.assertEqual(run.call_args[0][0], ["rustc", "--print", "cfg", "-C", "target-cpu=native"])
        with mock.patch.object(tools, "_run", return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")):
            self.assertEqual(tools.target_features([]), "none")
        with mock.patch.object(tools, "_run", side_effect=OSError("no rustc")):
            self.assertEqual(tools.target_features([]), "unavailable")

    @unittest.skipIf(shutil.which("rustc") is None, "rustc not installed")
    def test_target_features_of_this_machine(self):
        import platform

        features = tools.target_features([])
        if platform.machine() in ("arm64", "aarch64"):
            self.assertEqual(features, "neon")
        elif platform.machine() in ("x86_64", "AMD64"):
            self.assertNotIn("fma", features)  # the portable x86-64 build
            self.assertIn("fma", tools.target_features(["-C", "target-feature=+fma"]))

    @unittest.skipIf(shutil.which("rustc") is None, "rustc not installed")
    def test_toolchain_records_every_variable_cargo_builds_with(self):
        env = without_build_vars()
        with mock.patch.dict("os.environ", env, clear=True):
            plain = tools.toolchain()
        self.assertEqual({k: plain[k] for k in ("rustflags", "cargo_build_rustflags", "cargo_profile_release")},
                         {"rustflags": "", "cargo_build_rustflags": "", "cargo_profile_release": ""})
        self.assertNotIn(plain["target_features"], ("", "unavailable"))
        extra = {"CARGO_BUILD_RUSTFLAGS": "-C debuginfo=1", "CARGO_PROFILE_RELEASE_LTO": "fat",
                 "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16"}
        with mock.patch.dict("os.environ", {**env, **extra}, clear=True):
            tc = tools.toolchain()
        self.assertEqual(tc["cargo_build_rustflags"], "-C debuginfo=1")
        self.assertEqual(tc["cargo_profile_release"], "CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16;CARGO_PROFILE_RELEASE_LTO=fat")
        self.assertNotEqual(tools.build_identity(tc), tools.build_identity(plain))
        self.assertEqual(set(tools.build_identity(tc)), set(tools.BUILD_FIELDS))
        self.assertNotIn("llvm", tools.build_identity(tc))

    def test_a_build_for_the_compiling_cpu_is_refused(self):
        import tempfile

        self.assertTrue(tools.native_cpu(["-C", "target-cpu=native"]))
        self.assertTrue(tools.native_cpu(["-Ctarget-cpu=native", "-C", "opt-level=3"]))
        self.assertFalse(tools.native_cpu(["-C", "target-cpu=znver2"]))
        self.assertFalse(tools.native_cpu([]))
        with tempfile.TemporaryDirectory() as home:
            env = {**without_build_vars(), "CARGO_HOME": home}
            for var, value in (("RUSTFLAGS", "-C target-cpu=native"), ("CARGO_BUILD_RUSTFLAGS", "-Ctarget-cpu=native")):
                with self.subTest(var=var), mock.patch.dict("os.environ", {**env, var: value}, clear=True):
                    with self.assertRaisesRegex(tools.ToolError, "target-cpu=native"):
                        tools.toolchain()
            # RUSTFLAGS, set, overrides CARGO_BUILD_RUSTFLAGS: that build is not for the native CPU
            with mock.patch.dict("os.environ", {**env, "RUSTFLAGS": "-C target-cpu=znver2",
                                                "CARGO_BUILD_RUSTFLAGS": "-C target-cpu=native"}, clear=True):
                self.assertEqual(tools.toolchain()["rustflags"], "-C target-cpu=znver2")

    def test_cargo_configuration_that_changes_the_build_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d).resolve()
            work, home = d / "checkout" / "sub", d / "cargo-home"
            work.mkdir(parents=True)
            (d / "checkout" / ".cargo").mkdir()
            home.mkdir()
            comments = "# Keep default rustc CPU target.\n#   RUSTFLAGS=\"-C target-cpu=native\" cargo build\n"
            (d / "checkout" / ".cargo" / "config.toml").write_text(comments + "[net]\ngit-fetch-with-cli = true\n")
            env = {**without_build_vars(), "CARGO_HOME": str(home)}
            self.assertEqual(tools.cargo_config_files(work, env), [d / "checkout" / ".cargo" / "config.toml"])
            self.assertEqual(tools.cargo_config_problems(work, env), [])
            (home / "config.toml").write_text('[target.x86_64-unknown-linux-gnu]\nrustflags = ["-C", "target-cpu=znver4"]\n'
                                              "[profile.release]\nlto = true\n")
            (d / "checkout" / ".cargo" / "config").write_text('[build]\nrustflags = "-C target-feature=+fma"\n')
            problems = tools.cargo_config_problems(work, env)
            self.assertEqual(problems, [f"{d / 'checkout' / '.cargo' / 'config'}: build.rustflags",
                                        f"{home / 'config.toml'}: target.x86_64-unknown-linux-gnu.rustflags, "
                                        "profile.release"])
            (home / "config.toml").write_text("[build\n")
            self.assertIn("not valid TOML", tools.cargo_config_settings(home / "config.toml")[0])
            with mock.patch.dict("os.environ", env, clear=True):
                with self.assertRaisesRegex(tools.ToolError, "build.rustflags"):
                    tools.toolchain(work)

    @unittest.skipUnless(in_git(), "not a git checkout")
    def test_both_psmc_rs_identities_carry_the_build(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rs02 = fake_program(d / "rs02" / "psmc-rs", "psmc-rs 0.2.0")
            rs01 = fake_program(d / "psmc-rs-0.1" / "target" / "release" / "psmc-rs", "")
            dirs = {"base": d / "psmc-rs-0.1", "src": d / "psmc-rs-0.1" / "src", "target": d / "psmc-rs-0.1" / "target"}
            with mock.patch.object(tools, "rs02_binary", return_value=rs02), \
                    mock.patch.object(tools, "rs01_dirs", return_value=dirs), \
                    mock.patch.dict("os.environ", without_build_vars(), clear=True):
                t02, t01 = tools.psmc_rs_02(build=False), tools.psmc_rs_01(build=False)
                with mock.patch.dict("os.environ", {"CARGO_PROFILE_RELEASE_OPT_LEVEL": "2"}):
                    t02b, t01b = tools.psmc_rs_02(build=False), tools.psmc_rs_01(build=False)
        for a, b in ((t02, t02b), (t01, t01b)):
            self.assertTrue(set(tools.BUILD_FIELDS) <= set(a.identity), a.name)
            self.assertNotEqual(a.identity, b.identity, a.name)
            self.assertEqual(b.identity["cargo_profile_release"], "CARGO_PROFILE_RELEASE_OPT_LEVEL=2")


C_SOURCES = {
    "main.c": '#include <stdio.h>\n#include "psmc.h"\nint main(void) { fprintf(stderr, "Version: 0.6.5-test\\n"); '
              "return PSMC_ONE + f_core(); }\n",
    "core.c": "int f_core(void) { return 0; }\n",
    "psmc.h": "#define PSMC_ONE 1\nint f_core(void);\n",
    "kseq.h": "#define KSEQ_ZERO 0\n",
    "Makefile": "CFLAGS= -g -Wall -O2 -Wno-unused-function\n",
    "utils/fq2psmcfa.c": '#include "kseq.h"\nint main(void) { return KSEQ_ZERO; }\n',
    "utils/splitfa.c": '#include "kseq.h"\n#include <math.h>\nint main(void) { return KSEQ_ZERO * (int)sqrt(4.0); }\n',
}
for _name in ("khmm.c", "kmin.c", "cli.c", "em.c", "aux.c"):
    C_SOURCES[_name] = f"int f_{_name[:-2]}(void) {{ return 0; }}\n"


@unittest.skipIf(shutil.which("cc") is None, "no C compiler")
class BuildC(unittest.TestCase):
    """``tools --build-c``: lh3/psmc's layout with stand-in sources (psmc prints a version, as lh3/psmc does)."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tools_dir = Path(self._tmp.name)
        self.src = tools.c_psmc_dir(self.tools_dir)
        for name, text in C_SOURCES.items():
            (self.src / name).parent.mkdir(parents=True, exist_ok=True)
            (self.src / name).write_text(text)

    def build(self, **kw):
        return tools.build_c(self.src, cc="cc", log=lambda m: None, **kw)

    def test_defaults(self):
        import os

        env = {k: v for k, v in os.environ.items() if k not in ("CC", "PSMC_C_CFLAGS")}
        with mock.patch.dict("os.environ", env, clear=True):
            with mock.patch.object(tools.sys, "platform", "linux"), \
                    mock.patch.object(tools.platform, "machine", return_value="x86_64"):
                self.assertEqual(tools.default_cflags(), "-g -Wall -O2 -Wno-unused-function")  # the Makefile's
            with mock.patch.object(tools.sys, "platform", "darwin"), \
                    mock.patch.object(tools.platform, "machine", return_value="arm64"):
                self.assertEqual(tools.default_cflags(), "-g -Wall -O1 -Wno-unused-function")  # -O2 segfaults there
            with mock.patch.object(tools.sys, "platform", "darwin"), \
                    mock.patch.object(tools.platform, "machine", return_value="x86_64"):
                self.assertEqual(tools.default_cflags(), "-g -Wall -O2 -Wno-unused-function")
            self.assertEqual(tools.default_cc(), "gcc" if shutil.which("gcc") else "cc")
        with mock.patch.dict("os.environ", {"CC": "clang", "PSMC_C_CFLAGS": "-g -Wall -O1 -Wno-unused-function"}):
            self.assertEqual((tools.default_cc(), tools.default_cflags()), ("clang", "-g -Wall -O1 -Wno-unused-function"))

    def test_the_build_is_recorded_and_enters_c_psmcs_identity(self):
        record = self.build(cflags="-O1 -Wall", ldflags="")
        saved = json.loads((self.src / "BUILD.json").read_text())
        self.assertEqual(saved, record)
        self.assertEqual(record["cflags"], "-O1 -Wall")
        self.assertEqual(record["compiler"], shutil.which("cc"))
        self.assertTrue(record["compiler_version"])
        self.assertEqual(record["libs"], "-lm -lz")
        self.assertEqual(set(record["sources"]), set(C_SOURCES))
        self.assertEqual(record["sources"]["utils/splitfa.c"], tools.sha256_file(self.src / "utils" / "splitfa.c"))
        self.assertEqual(set(record["binaries"]), {"psmc", "utils/fq2psmcfa", "utils/splitfa"})
        for name, digest in record["binaries"].items():
            self.assertEqual(digest, tools.sha256_file(self.src / name), name)
        self.assertIn("-o psmc khmm.c kmin.c cli.c core.c em.c aux.c main.c -lm -lz", record["commands"]["psmc"])
        self.assertIn("-O1 -Wall -I.. -o fq2psmcfa fq2psmcfa.c", record["commands"]["utils/fq2psmcfa"])
        self.assertTrue(record["platform"])
        self.assertEqual([p.name for p in self.src.iterdir() if p.name.startswith(".")], [])  # nothing left over

        c = tools.c_psmc(self.tools_dir)
        self.assertEqual(c.identity, {"version": "0.6.5-test", "sha256": record["binaries"]["psmc"],
                                      "cc": record["compiler_version"], "cflags": "-O1 -Wall"})
        self.assertEqual(c.info["cc_path"], record["compiler"])
        self.assertEqual(tools.utility("splitfa", self.tools_dir).identity, {"sha256": record["binaries"]["utils/splitfa"]})
        self.build(cflags="-O2 -Wall")
        self.assertNotEqual(tools.c_psmc(self.tools_dir).identity["cflags"], c.identity["cflags"])

    def test_without_a_record_the_identity_is_as_before(self):
        self.build(cflags="-O1")
        (self.src / "BUILD.json").unlink()
        c = tools.c_psmc(self.tools_dir)
        self.assertEqual(set(c.identity), {"version", "sha256"})
        self.assertEqual(c.info, {})
        self.assertIsNone(tools.read_c_build(self.src))

    def test_a_binary_the_record_does_not_describe_is_refused(self):
        self.build(cflags="-O1")
        with open(self.src / "psmc", "ab") as f:
            f.write(b"\0")
        with self.assertRaisesRegex(tools.ToolError, "--build-c"):
            tools.c_psmc(self.tools_dir)

    def test_programs_built_otherwise_are_replaced_only_when_forced_and_kept(self):
        first = self.build(cflags="-O1")
        (self.src / "BUILD.json").unlink()  # as the laptop's, built by hand
        with self.assertRaisesRegex(tools.ToolError, "not described by .*--force"):
            self.build(cflags="-O2")
        for name, digest in first["binaries"].items():
            self.assertEqual(tools.sha256_file(self.src / name), digest, name)
            self.assertFalse((self.src / f"{name}.prev").exists())
        second = self.build(cflags="-O2", force=True)
        for name, digest in first["binaries"].items():
            self.assertEqual(tools.sha256_file(self.src / f"{name}.prev"), digest, name)
        self.assertFalse((self.src / "BUILD.json.prev").exists())
        # A build that BUILD.json describes is replaced without --force, and kept with its record.
        self.build(cflags="-O1 -Wall")
        self.assertEqual(json.loads((self.src / "BUILD.json.prev").read_text()), second)
        self.assertEqual(tools.sha256_file(self.src / "psmc.prev"), second["binaries"]["psmc"])
        self.assertEqual(tools.c_psmc(self.tools_dir).identity["cflags"], "-O1 -Wall")

    def test_a_compiler_that_hangs_is_a_tool_error(self):
        version = subprocess.CompletedProcess([], 0, stdout="cc 1.0\n", stderr="")
        with mock.patch.object(tools, "_run", side_effect=[version, subprocess.TimeoutExpired("cc", 600)]):
            with self.assertRaisesRegex(tools.ToolError, "building psmc timed out"):
                self.build()
        with mock.patch.object(tools, "_run", side_effect=subprocess.TimeoutExpired("cc", 60)):
            with self.assertRaisesRegex(tools.ToolError, "--version timed out"):
                self.build()
        self.assertFalse((self.src / "psmc").exists())

    def test_a_failed_build_keeps_the_old_programs_and_record(self):
        record = self.build(cflags="-O1")
        (self.src / "utils" / "splitfa.c").write_text("int main(void) { return }\n")
        with self.assertRaises(tools.ToolError):
            self.build(cflags="-O2")
        self.assertEqual(json.loads((self.src / "BUILD.json").read_text()), record)
        for name, digest in record["binaries"].items():
            self.assertEqual(tools.sha256_file(self.src / name), digest, name)
        with self.assertRaisesRegex(tools.ToolError, "sources missing"):
            (self.src / "em.c").unlink()
            self.build()
        with self.assertRaisesRegex(tools.ToolError, "not found"):
            tools.build_c(self.src, cc="no-such-compiler-here", log=lambda m: None)

    def test_the_tools_command_builds_first(self):
        from experiment.pipeline import __main__ as cli

        args = cli.build_parser().parse_args(["tools", "--build-c"])
        self.assertEqual((args.build_c, args.force), (True, False))
        self.assertFalse(cli.build_parser().parse_args(["tools"]).build_c)
        self.assertTrue(cli.build_parser().parse_args(["tools", "--build-c", "--force"]).force)
        missing = tools.ToolError("not here")
        with mock.patch.object(tools, "build_c", side_effect=tools.ToolError("cc failed")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.cmd_tools(args), 1)
        self.assertIn("cc failed", err.getvalue())
        with mock.patch.object(tools, "build_c") as build, \
                mock.patch.object(tools, "psmc_rs_02", side_effect=missing), \
                mock.patch.object(tools, "psmc_rs_01", side_effect=missing), \
                mock.patch.object(tools, "c_psmc", side_effect=missing), \
                mock.patch.object(tools, "utility", side_effect=missing), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.cmd_tools(args), 0)
        build.assert_called_once_with(force=False)


class RequirementsLock(unittest.TestCase):
    def test_the_lock_pins_what_the_pipeline_imports(self):
        lock = tools.read_lock()
        for name in ("msprime", "tskit", "numpy", "stdpopsim", "scipy", "matplotlib", "pandas", "psutil", "demes"):
            self.assertRegex(lock[name], r"^[0-9][0-9A-Za-z.+-]*$", name)
        self.assertIn("python", lock)

    def test_differences_are_named(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / "requirements.lock"
            lock.write_text("# python==0.0.1\nnumpy==0.0.1\nno-such-package-here==1.0\n")
            diffs = tools.lock_differences(lock)
        self.assertEqual(len(diffs), 3)
        self.assertTrue(any(d.startswith("no-such-package-here missing") for d in diffs))


if __name__ == "__main__":
    unittest.main()
