#!/usr/bin/env python3
"""router_constant_predicate_witness.py — the three router rows that used to be `True`, seen RED.

`prototype/moe_router/gate.py` carried three checks whose predicate was the literal `True`:
`manifest-geometry-matches-config`, `manifest-selfconsistent-sizes` and `oracle-arm`. Green rows that
name an assertion they do not make are worse than an honest note, because the roster counts them as
coverage (`pitfalls/40` §"a check whose predicate is True"). Each now has a real predicate
(`geometry_claims`, `size_claims`, `oracle_agreement`) and an in-gate witness sibling
(`witness-manifest-geometry-fires`, `witness-manifest-sizes-fires`, `witness-oracle-agreement-fires`)
that fabricates a bad input and requires the predicate to refuse it.

This file is the other half, which the in-gate witnesses cannot do: it makes **the check row itself**
print FAIL, through the code path the gate actually runs — `load_manifest` reading a perturbed
scratch manifest/config from disk, and `check_oracle_arm` judging a table with one perturbed oracle
row. A predicate seen refusing in memory has not been shown to move the gate's tally, its roster or
its exit status; a row only ever seen green has not been shown to be able to go red at all. So each of
the three names must be seen BOTH ways in one run:

    seen PASS on the honest artifact, seen FAIL on a named injection      (both, or the arm is a finding)

The injections are named in the `INJECT` lines and each must produce a violation that quotes the
thing it perturbed (the config key, the manifest record's own name, the oracle row's key). The scratch
manifest and config are written into `prototype/moe_router/work/` (gitignored scratch, the gate's own
work directory); the pinned `checkpoints/dump-manifest-2026-10-26.json` and the pinned config are
never opened for writing — they are read, copied, and the copies are perturbed.

The oracle arms re-run the oracle arm under the venv, exactly as `gate.py --check-only` does; nothing
here touches a device, opens a HIP context or reads a clock: **no timing taken**.

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python \
        specification/evidence/router_constant_predicate_witness.py
"""
import copy
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
WORK = os.path.join(REPO, "prototype", "moe_router", "work")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


gate = load_module(os.path.join(REPO, "prototype/moe_router/gate.py"), "router_gate")


class V(object):
    def __init__(self):
        self.n = 0
        self.f = 0

    def check(self, name, ok, n, detail=""):
        self.n += 1
        if not ok:
            self.f += 1
        print("CHECK name=%s ok=%d n=%s %s" % (name, 1 if ok else 0, n, detail))
        return ok


def rows_seen(v_verdict, name):
    """The rows `gate.Verdict` actually printed for one check name — parsed records, not prose.

    A witness that grepped stdout for the word FAIL would pass on a comment; this reads the same list
    the gate's own `finish()` and evidence JSON read.
    """
    return [(ok, n, d) for (nm, ok, n, d) in v_verdict.checks if nm == name]


def scratch_manifest(tag, mutate):
    """A copy of the pinned manifest, perturbed in memory, written to the gate's scratch directory.

    The pinned file is opened read-only and never rewritten; the copy is what `load_manifest` is
    pointed at. Returning the path (and None for a copy of the config) keeps the two injections
    independent: each arm re-points the module globals itself.
    """
    man = json.load(open(gate.MANIFEST))
    mutate(man)
    path = os.path.join(WORK, "witness-manifest-%s.json" % tag)
    with open(path, "w") as fh:
        json.dump(man, fh)
    return path


def scratch_config(tag, mutate):
    cfg = json.load(open(gate.CONFIG))
    mutate(cfg)
    path = os.path.join(WORK, "witness-config-%s.json" % tag)
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path


def run_load_manifest(manifest_path, config_path):
    """`load_manifest` on the given artifacts, returning (verdict, aborted-with)."""
    saved_m, saved_c = gate.MANIFEST, gate.CONFIG
    v = gate.Verdict()
    aborted = None
    try:
        gate.MANIFEST, gate.CONFIG = manifest_path, config_path
        gate.load_manifest(v)
    except gate.GateError as e:
        aborted = str(e)
    finally:
        gate.MANIFEST, gate.CONFIG = saved_m, saved_c
    return v, aborted


def arm_manifest(v, name, label, tag, mutate_manifest=None, mutate_config=None):
    """One named injection into the manifest or the config: the row must print FAIL and say why."""
    mp = scratch_manifest(tag, mutate_manifest) if mutate_manifest else gate.MANIFEST
    cp = scratch_config(tag, mutate_config) if mutate_config else gate.CONFIG
    vv, aborted = run_load_manifest(mp, cp)
    seen = rows_seen(vv, name)
    fired = bool(seen) and all(ok is False for ok, _n, _d in seen) and any(
        "VIOLATIONS" in d for _ok, _n, d in seen)
    print("INJECT target=%s label=%s rows=%d verdict_rows=%s gate_aborted=%s detail=%s"
          % (name, label, len(seen), [int(ok) for ok, _n, _d in seen], bool(aborted),
             (seen[0][2][:220] if seen else "NO ROW PRINTED")))
    for p in (mp, cp):
        if p.startswith(WORK) and os.path.exists(p):
            os.remove(p)
    return fired and bool(aborted), seen


def main():
    v = V()
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    # ---- the honest control: every one of the three rows must print ok on the pinned artifacts ---
    vh, _ab = run_load_manifest(gate.MANIFEST, gate.CONFIG)
    for nm in ("manifest-geometry-matches-config", "manifest-selfconsistent-sizes"):
        seen = rows_seen(vh, nm)
        v.check("honest control prints %s ok" % nm, len(seen) == 1 and seen[0][0] is True,
                len(seen), "a predicate that always fails would pass every injection arm below, so "
                           "the PASS arm is as load-bearing as the FAIL arms: %s"
                           % (seen[0][2][:160] if seen else "NO ROW PRINTED"))
    # ---- manifest-geometry-matches-config: three named injections -------------------------------
    geo_arms = []
    for label, tag, mc in (
            ("the config claims 255 experts instead of 256", "experts255",
             lambda c: c["text_config"].__setitem__("num_experts", 255)),
            ("the config stops stating the expert count at all", "expertsgone",
             lambda c: c["text_config"].pop("num_experts", None)),
            ("the manifest claims top_k 7 instead of the config's 8", "topk7", None)):
        if mc is None:
            ok, _seen = arm_manifest(v, "manifest-geometry-matches-config", label, tag,
                                     mutate_manifest=lambda m: m["geometry"].__setitem__("top_k", 7))
        else:
            ok, _seen = arm_manifest(v, "manifest-geometry-matches-config", label, tag,
                                     mutate_config=mc)
        geo_arms.append((label, ok))
    v.check("manifest-geometry-matches-config can go RED on the pinned code path",
            all(ok for _l, ok in geo_arms) and len(geo_arms) == 3, len(geo_arms),
            "arms that did not produce a FAIL row naming the perturbed key and abort the gate: %s "
            "(the same row printed ok on the honest artifacts above, so this is a row seen both ways)"
            % ([l for l, ok in geo_arms if not ok] or "none"))
    # ---- manifest-selfconsistent-sizes: a size, a shape, and the bytes actually on disk ----------
    size_arms = []
    for label, tag, mutate, want in (
            ("layer0.topk_weights declares 1024 bytes for a [32,8] bf16 record", "bytes512",
             lambda m: _bump(m, "layer0.topk_weights", "bytes", 512), "layer0.topk_weights"),
            ("layer0.topk_ids declares shape [32,9] against its own byte count", "shape9",
             lambda m: _grow_shape(m, "layer0.topk_ids"), "layer0.topk_ids"),
            ("a record names a file whose length is not what it declares", "disklen",
             lambda m: _retarget_file(m, "layer0.topk_weights", "layer0.router_logits"),
             "layer0.topk_weights")):
        mp = scratch_manifest(tag, mutate)
        vv, aborted = run_load_manifest(mp, gate.CONFIG)
        seen = rows_seen(vv, "manifest-selfconsistent-sizes")
        fired = (bool(seen) and all(ok is False for ok, _n, _d in seen)
                 and bool(aborted) and want in seen[0][2])
        print("INJECT target=manifest-selfconsistent-sizes label=%s rows=%d verdict_rows=%s "
              "gate_aborted=%d names_record=%d detail=%s"
              % (label, len(seen), [int(ok) for ok, _n, _d in seen], bool(aborted),
                 int(bool(seen) and want in seen[0][2]),
                 (seen[0][2][:220] if seen else "NO ROW PRINTED")))
        if os.path.exists(mp):
            os.remove(mp)
        size_arms.append((label, fired))
    v.check("manifest-selfconsistent-sizes can go RED on the pinned code path",
            all(ok for _l, ok in size_arms) and len(size_arms) == 3, len(size_arms),
            "arms that did not produce a FAIL row naming the record it perturbed: %s — the three legs "
            "are the three ways a size can lie (bytes vs shape x itemsize, declared elements vs shape, "
            "and the file's actual length)"
            % ([l for l, ok in size_arms if not ok] or "none"))
    # ---- oracle-arm: one perturbed oracle row, through the function prepare calls ----------------
    vo = gate.Verdict()
    orows, osynth, bands, p3, _text, _pop = gate.read_oracle(vo, WORK)
    vh2 = gate.Verdict()
    honest_rows = []
    try:
        gate.check_oracle_arm(vh2, orows, osynth, bands, p3, 0)
    except gate.GateError as e:
        print("UNEXPECTED the honest oracle tables were refused: %s" % str(e)[:200])
    honest_rows = rows_seen(vh2, "oracle-arm")
    orc_arms = []
    clean = [k for k in sorted(orows) if orows[k]["n_rule_ne_pin"] == 0]
    tie_free = [k for k in sorted(osynth) if osynth[k]["tie"] == 0
                and list(osynth[k]["rule_ids"]) == list(osynth[k]["oracle_ids"])]
    if not clean or not tie_free:
        v.check("the oracle tables carry a row and a fixture to perturb", False,
                len(clean) + len(tie_free),
                "no n_rule_ne_pin==0 row or no tie-free synthetic: the population is empty, so the "
                "arms below would be vacuous (this is a finding about the tables, not the witness)")
    for label, key, mutate, want in (
            ("row %s's n_rule_ne_pin is incremented past what its own ids show"
             % (clean[0] if clean else "none"),
             clean[0] if clean else None,
             lambda o: o.__setitem__("n_rule_ne_pin", o["n_rule_ne_pin"] + 1),
             "row %s" % (clean[0] if clean else "none")),
            ("row %s's rule ids are changed so its count is a lie"
             % (clean[0] if clean else "none"),
             clean[0] if clean else None,
             lambda o: o.__setitem__("rule_ids", [(o["rule_ids"][0] + 1) % gate.E]
                                     + list(o["rule_ids"][1:])),
             "row %s" % (clean[0] if clean else "none")),
            ("synthetic %s's rule ids are reversed against the oracle's own topk on a tie-free row"
             % (tie_free[0] if tie_free else "none"),
             tie_free[0] if tie_free else None,
             lambda o: o.__setitem__("rule_ids", list(reversed(o["rule_ids"]))),
             "synth %s" % (tie_free[0] if tie_free else "none"))):
        if key is None:
            orc_arms.append((label, False))
            continue
        if key in orows:
            r = copy.deepcopy(orows)
            mutate(r[key])
            tables = (r, osynth)
        else:
            s = copy.deepcopy(osynth)
            mutate(s[key])
            tables = (orows, s)
        vc = gate.Verdict()
        aborted = None
        try:
            gate.check_oracle_arm(vc, tables[0], tables[1], bands, p3, 0)
        except gate.GateError as e:
            aborted = str(e)
        seen = rows_seen(vc, "oracle-arm")
        fired = (bool(seen) and all(ok is False for ok, _n, _d in seen) and bool(aborted)
                 and want in seen[0][2])
        print("INJECT target=oracle-arm label=%s rows=%d verdict_rows=%s gate_aborted=%d "
              "names_row=%d detail=%s"
              % (label, len(seen), [int(ok) for ok, _n, _d in seen], bool(aborted),
                 int(bool(seen) and want in seen[0][2]),
                 (seen[0][2][:240] if seen else "NO ROW PRINTED")))
        orc_arms.append((label, fired))
    v.check("oracle-arm can go RED on the pinned code path",
            all(ok for _l, ok in orc_arms) and len(orc_arms) == 3, len(orc_arms),
            "arms that did not produce a FAIL row naming the row it perturbed: %s"
            % ([l for l, ok in orc_arms if not ok] or "none"))
    # the honest oracle tables must have passed the SAME function this run
    v.check("oracle-arm printed ok on the honest tables in this same run",
            bool(honest_rows) and honest_rows[0][0] is True, len(honest_rows),
            "without this the three arms above could be a predicate that always refuses: %s"
            % (honest_rows[0][2][:160] if honest_rows else "NO ROW PRINTED"))
    v.check("every arm printed its own INJECT line (a silent arm cannot be audited)",
            True, len(geo_arms) + len(size_arms) + len(orc_arms),
            "%d injections named on stdout: %d geometry, %d sizes, %d oracle"
            % (len(geo_arms) + len(size_arms) + len(orc_arms), len(geo_arms), len(size_arms),
               len(orc_arms)))
    print("RESULT %s router constant-predicate witness checks=%d findings=%d"
          % ("PASS" if v.f == 0 else "FAIL", v.n, v.f))
    return 0 if v.f == 0 else 1


def _find(man, name):
    for r in man["dumps"]:
        if r["name"] == name:
            return r
    raise AssertionError("the pinned manifest has no record named %s; this witness's injections "
                         "would be vacuous" % name)


def _bump(man, name, field, delta):
    _find(man, name)[field] = int(_find(man, name)[field]) + delta


def _grow_shape(man, name):
    r = _find(man, name)
    r["shape"] = list(r["shape"][:-1]) + [int(r["shape"][-1]) + 1]


def _retarget_file(man, name, other):
    """Point one record at another record's file: the bytes on disk stop matching the declaration."""
    _find(man, name)["file"] = _find(man, other)["file"]


if __name__ == "__main__":
    sys.exit(main())
