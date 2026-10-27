#!/usr/bin/env python3
"""One-off numeric probe over dumped .bin rows (development aid, not a gate).

Reads a handful of named rows out of a manifest and prints range/sum statistics so a human can
decide whether the bytes are the values the name claims. Deliberately prints no timing.
"""
import json
import os
import struct
import sys

def rows_for(manifest, names):
    m = json.load(open(manifest))
    out = {}
    for r in m["dumps"]:
        if r["name"] in names and r.get("written") and r["dtype"] == "f32":
            out.setdefault(r["name"], []).append(r)
    return out, m

def stats(path, dtype, shape):
    with open(path, "rb") as f:
        buf = f.read()
    if dtype == "f32":
        n = len(buf) // 4
        vals = struct.unpack("<%df" % n, buf[:4 * n])
    elif dtype == "bf16":
        n = len(buf) // 2
        raw = struct.unpack("<%dH" % n, buf[:2 * n])
        vals = [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in raw]
    else:
        return None
    return n, min(vals), max(vals), sum(vals)

def digests(manifest, names, want_graph):
    """per-row digest list for a name, so a repeated digest across blocks is visible by eye"""
    out, m = rows_for(manifest, set(names))
    print(f"manifest={os.path.basename(manifest)} gguf={m['gguf']['sha256'][:16]}")
    for nm in names:
        rs = [r for r in out.get(nm, []) if want_graph is None or r["graph"] == want_graph]
        seen = {}
        for r in rs:
            seen.setdefault(r["sha256"], []).append(r["block"])
        print(f"-- {nm}: {len(rs)} row(s), {len(seen)} distinct digest(s)")
        for d, blks in sorted(seen.items(), key=lambda kv: -len(kv[1]))[:6]:
            print(f"     {d[:16]} x{len(blks)} blocks={blks[:12]}")

def same_diff(ma, mb, names, want_graph):
    """per-block digest equality between two manifests: which blocks' values moved between runs"""
    a = json.load(open(ma))
    b = json.load(open(mb))
    for nm in names:
        def pick(m):
            d = {}
            for r in m["dumps"]:
                if r["name"] != nm or not r.get("written"):
                    continue
                if want_graph is not None and r["graph"] != want_graph:
                    continue
                d[(r["graph"], r["block"], r["seq"])] = r["sha256"]
            return d
        A, B = pick(a), pick(b)
        common = sorted(set(A) & set(B))
        same = [k for k in common if A[k] == B[k]]
        diff = [k for k in common if A[k] != B[k]]
        print(f"-- {nm}: compared={len(common)} same={len(same)} diff={len(diff)}")
        print(f"   first same keys={same[:8]}")
        print(f"   first diff keys={diff[:8]}")

def graphdiff(ma, mb):
    """per-graph written/compared/differing counts between two manifests, for the control write-up"""
    a = json.load(open(ma))
    b = json.load(open(mb))
    def written(m):
        d = {}
        for r in m["dumps"]:
            if r.get("written"):
                d[(r["graph"], r["block"], r["name"], r["op"], r["dtype"], tuple(r["shape"]), r["seq"])] = r
        return d
    A, B = written(a), written(b)
    per = {}
    for k in sorted(set(A) | set(B)):
        g = k[0]
        p = per.setdefault(g, {"a": 0, "b": 0, "compared": 0, "differing": 0, "only_a": 0, "only_b": 0})
        if k in A:
            p["a"] += 1
        if k in B:
            p["b"] += 1
        if k in A and k in B:
            p["compared"] += 1
            if A[k]["sha256"] != B[k]["sha256"]:
                p["differing"] += 1
        elif k in A:
            p["only_a"] += 1
        else:
            p["only_b"] += 1
    print(f"A={os.path.basename(ma)}  B={os.path.basename(mb)}")
    for g in sorted(per):
        p = per[g]
        print(f"  graph={g} written_A={p['a']} written_B={p['b']} compared={p['compared']} "
              f"differing={p['differing']} same={p['compared'] - p['differing']} "
              f"only_in_A={p['only_a']} only_in_B={p['only_b']}")

def dump_names(manifest, want_graph, want_block):
    """the dumped rows of one (graph, block) in node order, verbatim names included"""
    m = json.load(open(manifest))
    rows = [r for r in m["dumps"]
            if r["graph"] == want_graph and r["block"] == want_block]
    rows.sort(key=lambda r: (r["node"], r["seq"]))
    print(f"manifest={os.path.basename(manifest)} graph={want_graph} block={want_block} "
          f"rows={len(rows)}")
    for r in rows:
        print(f"  node={r['node']:>4} class={r['class']:<9} op={r['op']:<22} "
              f"dtype={r['dtype']:<5} shape={r['shape']} seq={r['seq']} "
              f"written={int(bool(r.get('written')))} name={r['name']!r}")

def namelist(manifest, want_graph, want_block):
    """one line per dumped row of a block, 'class:name[op]' — for quoting verbatim in a report"""
    m = json.load(open(manifest))
    rows = [r for r in m["dumps"] if r["graph"] == want_graph and r["block"] == want_block]
    rows.sort(key=lambda r: (r["node"], r["seq"]))
    print(" ".join(f"{r['class']}:{r['name']}[{r['op']}]" for r in rows))

def main():
    if sys.argv[2] == "namelist":
        namelist(sys.argv[1], int(sys.argv[3]), int(sys.argv[4]))
        return
    if sys.argv[2] == "names":
        dump_names(sys.argv[1], int(sys.argv[3]), int(sys.argv[4]))
        return
    if sys.argv[2] == "digests":
        digests(sys.argv[1], sys.argv[3].split(","),
                int(sys.argv[4]) if len(sys.argv) > 4 else None)
        return
    if sys.argv[2] == "graphdiff":
        graphdiff(sys.argv[1], sys.argv[3])
        return
    if sys.argv[2] == "same-diff":
        same_diff(sys.argv[1], sys.argv[3], sys.argv[4].split(","),
                  int(sys.argv[5]) if len(sys.argv) > 5 else None)
        return
    manifest = sys.argv[1]
    names = sys.argv[2].split(",")
    want_graph = int(sys.argv[3]) if len(sys.argv) > 3 else None
    out, m = rows_for(manifest, set(names))
    print(f"manifest={os.path.basename(manifest)} gguf={m['gguf']['sha256'][:16]}")
    for nm in names:
        rs = out.get(nm, [])
        if want_graph is not None:
            rs = [r for r in rs if r["graph"] == want_graph]
        print(f"-- {nm}: {len(rs)} written row(s)")
        for r in rs[:6]:
            s = stats(r["file"], r["dtype"], r["shape"])
            if s is None:
                print(f"     graph={r['graph']} block={r['block']} dtype={r['dtype']} (no stats)")
                continue
            n, lo, hi, tot = s
            print(f"     graph={r['graph']} block={r['block']} shape={r['shape']} n={n} "
                  f"min={lo:.6g} max={hi:.6g} sum={tot:.6g}")

if __name__ == "__main__":
    main()
