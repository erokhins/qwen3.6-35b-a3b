#!/usr/bin/env python3
"""chat_render.py — the chat template and the special-token ids, read out of the pinned artifact.

Why this exists. Two rules make a renderer load-bearing here rather than cosmetic:

  * `AGENTS.md` §"Never print chat-template special tokens" — every agent on this box is served by a
    checkpoint whose stop list contains its own end-of-turn id, so a literal special printed into a
    tool result truncates that message. A tool that handles transcripts must mask them, and the mask
    must come from the artifact's own vocabulary, not from literals typed into a source file.
  * `specification/02-tokenization-and-embedding.md` §02.4 records that the two committed witnesses
    DISAGREE about the stop id (model config `eos_token_id` = 248044; the shipping GGUF's
    `tokenizer.ggml.eos_token_id` = 248046) and deliberately picks no winner. A renderer that quietly
    adopted one would be making a user decision (`DECISIONS.md`) inside a utility module, so this file
    prints both, names both sources, and adopts neither.

Derived, never typed: the Jinja template from `tokenizer.chat_template`; the vocabulary and which ids
are special from `tokenizer.ggml.tokens` + `tokenizer.ggml.token_type` (CONTROL == 3, the value copied
from the pinned checkout `engines/llama.cpp` @ d327bf1f `ggml/include/ggml.h`, fetched 2026-10-26);
bos/eos/padding from `tokenizer.ggml.*_token_id`; the config's own bos/eos from
`checkpoints/Qwen3.6-35B-A3B.config.json`; vocab size from BOTH the artifact array length and
`text_config.vocab_size`, which are required to agree.

Provenance: adapted 2026-10-26 from the 0.8B engine repo's chat renderer (looked at, copied in, never
cited — `pitfalls/60-docs-and-citations.md`). That version takes an HF `tokenizer` object and calls
`apply_chat_template(...)`, taking its special literals from `tokenizer.added_tokens_decoder`. None of
that survives: this box has NO HF tokenizer directory for `Qwen/Qwen3.6-35B-A3B` — `models/qwen36-src/`
is four weight shards of 26 with no `tokenizer.json`/`vocab.json`/`merges.txt` — so the only spelling
of the template and the only vocabulary on this machine are the ones inside the published GGUF. The
`--turn role:TEXT` convention (split on the FIRST colon, empty content refused) is kept deliberately:
a probe and a dumper must agree on how a turn is spelled, and that is a rule about code, not intent.

What it does NOT do: tokenise. BPE needs `tokenizer.ggml.merges` plus the `qwen35` pre-tokenizer, and
the ids llama.cpp and `transformers` both agree on come from running one of them — owned by
`tools/dump_reference.py` and the fixtures, not by a renderer. `--render` prints the transcript, its
length, a checksum and the specials it contains BY ID.

Run with the CPU venv (only for Jinja2; torch is never imported and no device context is created):

    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python tools/chat_render.py --inventory
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python tools/chat_render.py \\
        --turn user:What is the 8-of-256 rule? --render
    /home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python tools/chat_render.py --selftest

(`--selftest` is run under the venv too, because its clean control renders; the injected-defect cases
need no Jinja2 and pass under plain `python3` as well.)

CPU only: reads the header, prints no timing, touches no device (`pitfalls/00-measurement.md`).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_DEFAULT = os.path.join(REPO, "models/unsloth-q4km/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
CFG_DEFAULT = os.path.join(REPO, "checkpoints/Qwen3.6-35B-A3B.config.json")

TOKEN_TYPE_NORMAL, TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED, \
    TOKEN_TYPE_BYTE = 1, 2, 3, 4, 6
KIND_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
            11: "<q", 12: "<d"}
ROLES = ("user", "assistant", "system")


class Fatal(Exception):
    """Every refusal raises this. Nothing here prints a partial answer and exits 0."""


def die(fmt, *a):
    raise Fatal(fmt % a if a else fmt)


def read_kv(path):
    """Parse a GGUF's KV metadata section and stop. Tensor records and the data section belong to
    `tools/gguf_reader.py`, which gates them behind its selftest token; this opens no weight byte."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            die("%s is not a GGUF (no GGUF magic). The safetensors checkpoint spells its families "
                "differently — use tools/weights_report.py for that artifact.", path)
        (version,) = struct.unpack("<I", fh.read(4))
        (n_tensors,) = struct.unpack("<Q", fh.read(8))
        (n_kv,) = struct.unpack("<Q", fh.read(8))

        def string():
            (n,) = struct.unpack("<Q", fh.read(8))
            if n > (1 << 31):
                die("%s: a %d-byte string in the header is file data, not metadata", path, n)
            return fh.read(n).decode("utf-8")

        def value(vt):
            if vt == 8:
                return string()
            if vt == 9:
                (et,) = struct.unpack("<I", fh.read(4))
                (n,) = struct.unpack("<Q", fh.read(8))
                if n > (1 << 25):
                    die("%s: an array of %d elements is file data, not metadata", path, n)
                return [value(et) for _ in range(n)]
            if vt not in KIND_FMT:
                die("%s: unknown GGUF value kind %d — guessing its width would shift every field "
                    "after it and the table would still print", path, vt)
            fmt = KIND_FMT[vt]
            return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]

        kv = {}
        for _ in range(n_kv):
            k = string()
            (vt,) = struct.unpack("<I", fh.read(4))
            kv[k] = value(vt)
    return kv, n_tensors, version


def load_artifact(gguf, config):
    kv, n_tensors, version = read_kv(gguf)
    arch = kv.get("general.architecture")
    if arch != "qwen35moe":
        die("%s declares general.architecture=%r; the key names and expectations here are for the "
            "shipping qwen35moe artifact", gguf, arch)
    for k in ("tokenizer.chat_template", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type"):
        if k not in kv:
            die("%s carries no %s. There is then nothing to render or mask with, and a renderer that "
                "invented one would produce a transcript no reference ever saw", gguf, k)
    tokens, types = kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.token_type"]
    if len(tokens) != len(types):
        die("%s: tokenizer.ggml.tokens has %d entries, token_type has %d. The vocabulary and its "
            "specialness map are not the same population, so no id-is-special answer is sound",
            gguf, len(tokens), len(types))
    t = json.load(open(config))["text_config"]
    if len(tokens) != t["vocab_size"]:
        die("%s: the artifact vocabulary is %d tokens, the config says vocab_size=%d. The two "
            "authorities on the embedding table's row count disagree", gguf, len(tokens),
            t["vocab_size"])
    specials = {i: tok for i, (tok, ty) in enumerate(zip(tokens, types))
                if ty == TOKEN_TYPE_CONTROL}
    if not specials:
        die("%s: no token has token_type CONTROL(%d), so nothing would be masked and every special "
            "would be printed literally (AGENTS.md)", gguf, TOKEN_TYPE_CONTROL)
    named, named_kind = {}, {}
    # bos and eos must be CONTROL: they are the delimiters that reach printed output, and a mask built
    # from CONTROL would leave them literal (the AGENTS.md defect). Padding is NOT held to that: llama.cpp
    # commonly types its pad as NORMAL, and a pad never appears in a rendered prompt, so demanding
    # CONTROL there would refuse a good artifact for a formatting choice that is not a defect.
    for key, must_control in (("tokenizer.ggml.bos_token_id", True),
                              ("tokenizer.ggml.eos_token_id", True),
                              ("tokenizer.ggml.padding_token_id", False)):
        if key not in kv:
            die("%s carries no %s", gguf, key)
        i = int(kv[key])
        if not 0 <= i < len(tokens):
            die("%s: %s = %d is outside the vocabulary [0, %d). Nothing downstream of that id — "
                "including the stop rule — can be trusted", gguf, key, i, len(tokens))
        named[key.rsplit(".", 1)[1]] = i
        named_kind[key.rsplit(".", 1)[1]] = types[i]
        if must_control and types[i] != TOKEN_TYPE_CONTROL:
            die("%s: %s points at id %d, which token_type calls %s, not CONTROL(%d). Either the "
                "metadata is inconsistent or the vocabulary was repacked; in both cases masking by "
                "CONTROL would leave this token literal in printed output — the exact defect "
                "AGENTS.md exists to prevent",
                gguf, key, i, types[i], TOKEN_TYPE_CONTROL)
    return dict(path=gguf, kv=kv, tokens=tokens, types=types, specials=specials, named=named,
                named_kind=named_kind, template=kv["tokenizer.chat_template"], vocab=len(tokens),
                n_tensors=n_tensors, version=version, cfg_text=t, arch=arch)


def mask_specials(text, art):
    """Every special in `text` replaced by `[id]`. Every string this tool prints goes through it."""
    for tid in sorted(art["specials"]):
        lit = art["specials"][tid]
        if lit and lit in text:
            text = text.replace(lit, "[%d]" % tid)
    return text


def template_specials(art):
    """(id, literal) for each CONTROL token the template actually uses, in id order."""
    used = [(i, lit) for i, lit in sorted(art["specials"].items()) if lit in art["template"]]
    if not used:
        die("%s: the chat template uses none of the %d CONTROL tokens in this vocabulary. A template "
            "with no turn delimiters is not a chat template for this model, and rendering with it "
            "would hand the reference a plain-text prompt it never saw", art["path"],
            len(art["specials"]))
    return used


def parse_turn(spec):
    """One `--turn` argument as a message dict; split on the FIRST colon so content may contain them."""
    role, sep, content = spec.partition(":")
    if not sep or role.strip() not in ROLES:
        die("--turn expects {user|assistant|system}:TEXT, got %r", spec)
    if content == "":
        die("--turn %r has an empty content; a real empty turn is legal but write it as a space so "
            "this cannot be a typo for a missing colon", spec)
    return {"role": role.strip(), "content": content}


def render(art, messages, thinking=False):
    """The transcript, rendered by the artifact's own template. Needs Jinja2 (the CPU venv has it)."""
    try:
        import jinja2
    except ImportError:
        die("rendering needs Jinja2 — run with the CPU oracle venv "
            "/home/erokhins/erokhins_projects/qwen3.5-0.8b/.venv/bin/python (this file never imports "
            "torch and creates no device context)")
    used = template_specials(art)
    env = jinja2.Environment(keep_trailing_newline=True)
    try:
        out = env.from_string(art["template"]).render(
            messages=list(messages), add_generation_prompt=True, enable_thinking=bool(thinking))
    except jinja2.TemplateError as e:
        die("%s: the embedded template failed to render (%s: %s). The prompt the reference would have "
            "been given cannot be reproduced, so nothing here is pinable", art["path"],
            type(e).__name__, " ".join(str(e).split())[:160])
    if not out.strip():
        die("%s: the template rendered to nothing for %d message(s); a check over an empty prompt "
            "cannot fail, so this is an error, not a pass", art["path"], len(messages))
    if not any(lit in out for _, lit in used):
        die("%s: the render contains none of the delimiters the template uses (%s); the template is "
            "not doing what its name says", art["path"], ", ".join(str(i) for i, _ in used))
    return out, used


def inventory(art, out):
    out("== artifact")
    out("   file                       %s" % os.path.relpath(art["path"], REPO))
    out("   gguf version / n_tensors   %d / %d" % (art["version"], art["n_tensors"]))
    out("   architecture               %s" % art["arch"])
    out("   vocab (tokens)             %d == config text_config.vocab_size  OK" % art["vocab"])
    out("== special tokens, from tokenizer.ggml.token_type == CONTROL(%d)" % TOKEN_TYPE_CONTROL)
    for tid in sorted(art["specials"]):
        tag = "".join("  <- %s" % k for k, v in sorted(art["named"].items()) if v == tid)
        out("   id %-8d %s  token_type=%s%s"
            % (tid, mask_specials(art["specials"][tid], art), TOKEN_TYPE_CONTROL, tag))
    for k, v in sorted(art["named"].items()):
        if v not in art["specials"]:
            out("   id %-8d (named %s) is token_type=%d, not CONTROL — legal for a pad, which never "
                "appears in a rendered prompt" % (v, k, art["named_kind"][k]))
    out("== the stop id the two committed witnesses disagree about (specification/02 §02.4)")
    out("   config text_config.bos_token_id      %d" % art["cfg_text"]["bos_token_id"])
    out("   config text_config.eos_token_id      %d" % art["cfg_text"]["eos_token_id"])
    out("   gguf   tokenizer.ggml.bos_token_id   %d" % art["named"]["bos_token_id"])
    out("   gguf   tokenizer.ggml.eos_token_id   %d" % art["named"]["eos_token_id"])
    out("   NOT DECIDED here. Which id the engine stops on is a user decision (DECISIONS.md), not a")
    out("   renderer's: both are printed, neither adopted, and no literal is printed at all.")
    used = template_specials(art)
    out("== the template uses %d of those %d CONTROL tokens: %s"
        % (len(used), len(art["specials"]), ", ".join(str(i) for i, _ in used)))
    out("   template length            %d chars, sha256 %s…"
        % (len(art["template"]), hashlib.sha256(art["template"].encode()).hexdigest()[:12]))
    if len(used) < len(art["specials"]):
        out("   note: %d CONTROL tokens are unused by the template (the vision placeholders and the "
            "think-boundary pair are the expected ones; vision is out of scope, DECISIONS.md D-04)"
            % (len(art["specials"]) - len(used)))


# ---------------------------------------------------------------------------------------------
# selftest — synthetic GGUFs in tmp/, one defect each, each refused by name.
# ---------------------------------------------------------------------------------------------
def write_gguf(path, kv):
    """Write a minimal but FORMAT-correct GGUF header (no tensors), for the selftest only.

    Correctness of this writer is what makes the selftest mean anything: the reader under test is
    `read_kv`, so a writer that disagreed with the format in the same way would prove nothing. Kinds
    and layouts are the ones `read_kv` implements (bare string = u64 length + bytes; array = element
    kind u32 + count u64 + bare elements; scalars = the `KIND_FMT` width).
    """

    def bare(v):
        if isinstance(v, bool):
            return struct.pack("<?", v), 7
        if isinstance(v, str):
            b = v.encode()
            return struct.pack("<Q", len(b)) + b, 8
        if isinstance(v, list):
            out, et = b"", 4
            for x in v:
                body, et = bare(x)
                out += body
            return struct.pack("<I", et) + struct.pack("<Q", len(v)) + out, 9
        return struct.pack("<I", int(v)), 4

    body = b""
    for k, v in kv.items():
        kb, _ = bare(k)
        body += kb                                   # keys are bare strings, no kind prefix
        vb, kind = bare(v)
        body += struct.pack("<I", kind) + vb          # values carry their kind
    with open(path, "wb") as fh:
        fh.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1)
                 + struct.pack("<Q", len(kv)) + body + b"tail-not-read")


START, END = "«im_start»", "«im_end»"


def synth_kv():
    vocab = 12
    tokens = ["tok%d" % i for i in range(vocab)]
    types = [TOKEN_TYPE_NORMAL] * vocab
    for i, lit in ((10, START), (11, END)):
        tokens[i], types[i] = lit, TOKEN_TYPE_CONTROL
    tmpl = ("{% for m in messages %}" + START + "{{ m.role }}\n{{ m.content }}" + END +
            "\n{% endfor %}{% if add_generation_prompt %}" + START + "assistant\n{% endif %}")
    return {"general.architecture": "qwen35moe",
            "tokenizer.chat_template": tmpl,
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.bos_token_id": 10,
            "tokenizer.ggml.eos_token_id": 11,
            "tokenizer.ggml.padding_token_id": 9}


def selftest(root):
    import shutil
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    cfg = json.load(open(CFG_DEFAULT))
    small = os.path.join(root, "config_small.json")
    c = json.loads(json.dumps(cfg))
    c["text_config"].update(vocab_size=12, bos_token_id=10, eos_token_id=11)
    json.dump(c, open(small, "w"))
    fired, missed = [], []

    def build(kv, name="case.gguf"):
        p = os.path.join(root, name)
        write_gguf(p, kv)
        return p

    def expect(label, mutate, needle, cfg_path=None):
        kv = synth_kv()
        mutate(kv)
        try:
            art = load_artifact(build(kv), cfg_path or small)
            inventory(art, lambda *a, **k: None)
            render(art, [{"role": "user", "content": "hi"}])
        except Fatal as e:
            msg = " ".join(str(e).split())
            if needle.lower() not in msg.lower():
                missed.append("%s (wrong reason: wanted %r, got …%s)" % (label, needle, msg[-110:]))
                print("  WRONG REASON %-42s …%s" % (label, msg[-110:]))
                return
            fired.append(label)
            print("  FIRED   %-46s …%s" % (label, msg[-108:]))
            return
        missed.append(label)
        print("  NOT FIRED %-44s the defect ran clean" % label)

    print("chat_render --selftest: synthetic GGUFs, one defect each")
    art = load_artifact(build(synth_kv(), "clean.gguf"), small)
    inventory(art, lambda *a, **k: None)
    out, used = render(art, [{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "there"},
                             {"role": "user", "content": "and the 8 of 256?"}])
    masked = mask_specials(out, art)
    for _, lit in used:
        if lit in masked:
            die("a special literal survived the mask")
    if "[%d]" % used[0][0] not in masked:
        die("the mask produced no id-form token, so it masked nothing and proved nothing")
    print("  CONTROL   clean artifact renders a 3-turn transcript (%d chars) and masks every special "
          "to [id]" % len(masked))

    expect("chat_template missing", lambda kv: kv.pop("tokenizer.chat_template"), "no tokenizer.chat_template")
    expect("tokens array missing", lambda kv: kv.pop("tokenizer.ggml.tokens"), "no tokenizer.ggml.tokens")
    expect("architecture is not qwen35moe", lambda kv: kv.__setitem__("general.architecture", "llama"),
           "general.architecture")
    expect("token_type array one short of the vocabulary",
           lambda kv: kv.__setitem__("tokenizer.ggml.token_type",
                                     kv["tokenizer.ggml.token_type"][:-1]), "not the same population")
    expect("eos id outside the vocabulary",
           lambda kv: kv.__setitem__("tokenizer.ggml.eos_token_id", 999), "outside the vocabulary")
    expect("eos id not marked CONTROL",
           lambda kv: kv.__setitem__("tokenizer.ggml.token_type",
                                     kv["tokenizer.ggml.token_type"][:11] + [TOKEN_TYPE_NORMAL]
                                     + kv["tokenizer.ggml.token_type"][12:]), "not CONTROL")
    expect("no CONTROL token at all",
           lambda kv: kv.__setitem__("tokenizer.ggml.token_type",
                                     [TOKEN_TYPE_NORMAL] * 12), "no token has token_type CONTROL")
    expect("template uses no CONTROL token",
           lambda kv: kv.__setitem__("tokenizer.chat_template",
                                     "{% for m in messages %}{{ m.content }}{% endfor %}"),
           "uses none of the")
    expect("template delimiters are not vocabulary entries",
           lambda kv: kv.__setitem__("tokenizer.chat_template",
                                     "{% for m in messages %}<|im_start|>{{ m.content }}{% endfor %}"),
           "uses none of the")
    expect("vocab disagrees with the config",
           lambda kv: None, "vocab_size", cfg_path=CFG_DEFAULT)
    expect("padding id outside the vocabulary",
           lambda kv: kv.__setitem__("tokenizer.ggml.padding_token_id", 4242), "outside the vocabulary")

    # CLI-level controls: a --turn with no colon, and an empty content, are refusals not surprises.
    for spec, needle in (("hello world", "expects"), ("user:", "empty content")):
        try:
            parse_turn(spec)
        except Fatal as e:
            if needle in " ".join(str(e).split()).lower():
                fired.append("--turn %r" % spec)
                print("  FIRED   %-46s …%s" % ("--turn %r" % spec,
                                               " ".join(str(e).split())[-100:]))
            else:
                missed.append("--turn %r (wrong reason)" % spec)
                print("  WRONG REASON %-42s %s" % ("--turn %r" % spec, e))
        else:
            missed.append("--turn %r" % spec)
            print("  NOT FIRED %-44s accepted a malformed turn" % ("--turn %r" % spec,))

    try:
        write_gguf(os.path.join(root, "not_gguf.bin"), {})
        with open(os.path.join(root, "not_gguf.bin"), "r+b") as fh:
            fh.write(b"XXXX")
        load_artifact(os.path.join(root, "not_gguf.bin"), small)
    except Fatal as e:
        if "not a gguf" in " ".join(str(e).split()).lower():
            fired.append("not a GGUF")
            print("  FIRED   %-46s …%s" % ("not a GGUF", " ".join(str(e).split())[-100:]))
        else:
            missed.append("not a GGUF (wrong reason)")
    else:
        missed.append("not a GGUF")

    if missed:
        print("\nSELFTEST FAIL: %d of %d faults did not fire for the right reason: %s"
              % (len(missed), len(missed) + len(fired), "; ".join(missed)))
        return 1
    print("\nSELFTEST PASS: %d injected defects each refused by name, and the clean artifact renders "
          "and masks" % len(fired))
    return 0


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", default=GGUF_DEFAULT)
    ap.add_argument("--config", default=CFG_DEFAULT)
    ap.add_argument("--turn", action="append", default=[],
                    help="role:TEXT, repeatable; the FIRST colon splits role from content")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--emit", help="write the MASKED render here (never the literals)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(os.path.join(REPO, "tmp", "w17_chat_selftest"))
    try:
        art = load_artifact(a.gguf, a.config)
        inventory(art, print)
        if a.render or a.emit:
            if not a.turn:
                die("--render/--emit needs at least one --turn role:TEXT; rendering an empty "
                    "transcript would certify a prompt that does not exist")
            msgs = [parse_turn(s) for s in a.turn]
            out, used = render(art, msgs, thinking=a.thinking)
            print("== render (%d messages, thinking=%s)" % (len(msgs), a.thinking))
            print("   chars %d, sha256 %s…" % (len(out), hashlib.sha256(out.encode()).hexdigest()[:16]))
            print("   delimiters present by id: %s"
                  % ", ".join(str(i) for i, lit in used if lit in out))
            print("   masked transcript (literals replaced by [id], per AGENTS.md):")
            for line in mask_specials(out, art).splitlines():
                print("   | %s" % line)
            if a.emit:
                with open(a.emit, "w") as fh:
                    fh.write(mask_specials(out, art))
                print("   wrote the MASKED render to %s (the unmasked text is never written to disk "
                      "by this tool: a pinned prompt is a file a later agent will cat)" % a.emit)
    except Fatal as e:
        print("REFUSED: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))

