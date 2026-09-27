#!/usr/bin/env python3
"""ragcli — the app layer: config file, ingestion, CLI, browser UI.

The engine lives in rag.py. This module is the part a non-technical user
touches:

    rag init     choose your model backend, write rag.json
    rag add      index files or folders (PDF, DOCX, markdown, text)
    rag ask      get a cited answer from your documents
    rag serve    browser UI
    rag doctor   when something doesn't work

Resolution order everywhere: CLI flag > env var > rag.json > default.
"""
import base64
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import rag
from rag import Rag, _api_base, recommend

CFG = {}
rag.CFG = CFG  # engine and app layer share one dict, so they never disagree

CONFIG_KEYS = (
    "llm_base", "llm_model", "llm_api_key", "llm_reasoning", "embed_base",
    "embed_model", "embed_max_chars", "db", "corpus", "glossary", "answer_mode",
    "serve_host", "serve_port",
)

# ponytail: four presets cover the realistic setups; a fifth just adds a menu
# entry nobody reads. Add one when a provider actually shows up in the wild.
PRESETS = {
    # local llama.cpp-style servers often run a reasoning model; measured 20.1s
    # -> 3.4s per rewrite with reasoning_effort="none" on this one.
    "local": {"llm_base": "http://127.0.0.1:8888", "llm_model": "", "llm_reasoning": "none"},
    "ollama": {"llm_base": "http://127.0.0.1:11434", "llm_model": "llama3.1:8b"},
    "openrouter": {"llm_base": "https://openrouter.ai/api", "llm_model": "openai/gpt-4o-mini"},
    "openai": {"llm_base": "https://api.openai.com", "llm_model": "gpt-4o-mini"},
}


def config_path(flag=None):
    return flag or os.environ.get("RAG_CONFIG", "rag.json")


def load_config(path=None):
    """Missing file -> {} (defaults apply). Malformed -> one friendly line, then {}."""
    path = config_path(path)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"warning: ignoring {path} ({e})")
        return {}
    if not isinstance(data, dict):
        print(f"warning: ignoring {path} (not a JSON object)")
        return {}
    return {k: v for k, v in data.items() if k in CONFIG_KEYS}


def resolve(name, flag, env, default):
    """CLI flag > env var > rag.json > default."""
    if flag is not None and flag != "":
        return flag
    if env and os.environ.get(env):
        return os.environ[env]
    v = CFG.get(name)
    return v if v not in (None, "") else default


# --------------------------------------------------------------------------- ingestion
TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".html", ".htm", ".log"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}
SKIP_FILES = {"rag.db", "rag.json", "glossary.txt"}

# --------------------------------------------------------------------------- vision
# Images become searchable text through the LLM's own vision capability — no OCR
# package, no extra dependency. The base64 payload is passed through llm() as an
# OpenAI content block (type: "image_url"); llm() forwards messages to the
# endpoint unmodified, so any vision model the user pointed it at already works.
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff",
}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB: enough for phone photos, small enough
                                       # to base64 into a single request
VISION_MAX_TOKENS = 2048              # long OCR plus a scene description
VISION_TIMEOUT = 120                  # seconds per vision call
MAX_VIDEO_FRAMES = 12                 # frames sampled from a video, evenly spaced
VISION_PROMPT = (
    "You transcribe and describe an image for a search index. First, transcribe all "
    "legible text exactly, keeping its wording and language. Then, in plain keywords, "
    "describe the visual content: objects, places, people, actions, and any text on "
    "signs or documents. Use only what is in the image; if there is no readable text, "
    "say 'no readable text' briefly and describe the scene. Do not add outside "
    "knowledge."
)


class ExtractError(Exception):
    pass


def extract(path):
    """File -> text. Extractors stay optional; a missing one names its install
    command instead of raising a traceback."""
    ext = Path(path).suffix.lower()
    if ext in TEXT_EXT:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        try:
            import pypdf
        except ImportError:
            raise ExtractError("PDF needs pypdf — run: uv add pypdf") from None
        try:
            return "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(path).pages)
        except Exception as e:  # a corrupt or encrypted PDF must not kill the batch
            raise ExtractError(f"could not read {path}: {e}") from None
    if ext == ".docx":
        try:
            import docx
        except ImportError:
            raise ExtractError("DOCX needs python-docx — run: uv add python-docx") from None
        try:
            return "\n".join(p.text for p in docx.Document(path).paragraphs)
        except Exception as e:
            raise ExtractError(f"could not read {path}: {e}") from None
    if ext in IMAGE_EXT:
        return describe_image(path)
    if ext in VIDEO_EXT:
        return describe_video(path)
    raise ExtractError(f"unsupported format: {ext or '(no extension)'}")


def describe_image(path):
    """Image -> text via the LLM's vision capability. Raises ExtractError with an
    actionable message when the LLM is unreachable or lacks vision support — the
    image is never silently dropped and never padded with invented text."""
    p = Path(path)
    size = p.stat().st_size
    if size == 0:
        raise ExtractError(f"image {p.name} is empty — skipping")
    if size > MAX_IMAGE_BYTES:
        mb = size / 1024 / 1024
        cap = MAX_IMAGE_BYTES / 1024 / 1024
        raise ExtractError(
            f"image {p.name} is {mb:.0f} MB (>{cap:.0f} MB vision cap) — too large; "
            "resize it before running rag add"
        )
    data = p.read_bytes()
    mime = IMAGE_MIME[p.suffix.lower()]
    messages = [
        {"role": "system", "content": VISION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Transcribe and describe this image for a search index."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"},
                },
            ],
        },
    ]
    text = rag.llm(messages, max_tokens=VISION_MAX_TOKENS, timeout=VISION_TIMEOUT)
    if not text:
        raise ExtractError(
            f"could not read image {p.name}: the LLM returned no text — your model may "
            "not support vision, or the endpoint is unreachable. Run: rag doctor"
        )
    return text.strip()


def _video_duration(path):
    """Seconds via ffprobe's metadata (fast — no decode). None if unknown or
    ffprobe is missing."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        text = (out.stdout or "").strip()
        return float(text) if text else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _frames_from_video(path, outd, n):
    """Extract ~n evenly spaced frames to outd with ffmpeg. Returns None when
    ffmpeg is missing, otherwise a list of frame paths (possibly empty). Frames
    are scaled to a 1280px width so each vision payload stays small."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None  # ffmpeg absent/hung — the caller raises the install hint
    dur = _video_duration(path)
    if dur and n > 1:
        fps = max((n - 1) / dur, 1.0 / 3600.0)  # even spread; never slower than ~1 frame/hour
    else:
        # No ffprobe duration available: sample from the start. A 2-minute listing
        # clip still yields 12 frames across the first ~24s; full even spacing needs
        # ffprobe (bundled with ffmpeg in any normal install).
        fps = 0.5
    vf = "fps=%s,scale=1280:-1" % round(fps, 4)
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vf", vf, "-frames:v", str(n), "-f", "image2", "-y",
             str(outd / "frame_%03d.png")],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return []
    return sorted(outd.glob("frame_*.png"))


def describe_video(path):
    """Video -> text: sample evenly spaced frames with ffmpeg and describe each via
    vision. Raises ExtractError (with the ffmpeg install hint) when frames can't be
    produced, or when the LLM can't read any frame."""
    p = Path(path)
    if p.stat().st_size == 0:
        raise ExtractError(f"video {p.name} is empty — skipping")
    print(f"[video] {p.name}: sampling up to {MAX_VIDEO_FRAMES} frames via ffmpeg…",
          file=sys.stdout)
    outd = Path(tempfile.mkdtemp(prefix="ragvid_"))
    try:
        frames = _frames_from_video(path, outd, MAX_VIDEO_FRAMES)
        if frames is None:
            raise ExtractError(
                f"video {p.name} needs ffmpeg to extract frames — install it "
                "(apt install ffmpeg / brew install ffmpeg / choco install ffmpeg) "
                "and re-run rag add"
            )
        if not frames:
            raise ExtractError(
                f"video {p.name}: ffmpeg produced no frames — unreadable file"
            )
        parts = []
        for f in frames:
            try:
                t = describe_image(str(f))
                if t:
                    parts.append(t)
            except ExtractError:
                pass
        if not parts:
            raise ExtractError(
                f"video {p.name}: the LLM returned no text for any frame — run rag "
                "doctor to check that the model supports vision"
            )
        return "\n\n".join(parts)
    finally:
        shutil.rmtree(outd, ignore_errors=True)


def walk(paths, corpus):
    """Recursive file list, skipping VCS/venv/cache dirs, hidden entries, db/config.

    The hidden/venv rule is applied to parts *below* the walked root, not the
    absolute prefix — otherwise a corpus living under any dot-directory (a
    cache dir, ~/.hermes, a CI workspace) would index as empty."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            if p.name not in SKIP_FILES:
                out.append(p)
            continue
        for f in sorted(p.rglob("*")):
            if not f.is_file() or f.name in SKIP_FILES:
                continue
            rel = f.relative_to(p) if f != p else Path("")
            if any(part.startswith(".") or part in SKIP_DIRS for part in rel.parts):
                continue
            out.append(f)
    return out


def doc_id(path, corpus):
    """Relative posix path — stable id, readable in citations."""
    try:
        return Path(path).resolve().relative_to(Path(corpus).resolve()).as_posix()
    except ValueError:
        return Path(path).name


# --------------------------------------------------------------------------- browser UI
PAGE = """<!doctype html>
<html><head><meta charset=utf-8><title>rag</title>
<style>body{font:16px system-ui;max-width:52em;margin:2em auto;padding:0 1em}
textarea{width:100%;font:inherit}button{font:inherit;padding:.4em 1em}
#a{white-space:pre-wrap;margin:1em 0}#c li{margin:.4em 0;color:#555}</style></head>
<body><h1>rag</h1><p id=stats></p>
<textarea id=q rows=2 placeholder="ask about your documents"></textarea>
<p><button onclick=ask()>Ask</button></p><div id=a></div><ol id=c></ol>
<script>
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function ask(){
  const q = document.getElementById('q').value;
  const r = await fetch('/api/ask?q=' + encodeURIComponent(q));
  const d = await r.json();
  document.getElementById('a').textContent = d.answer || d.note || '';
  document.getElementById('c').innerHTML = (d.citations||[]).map(
    x => `<li><b>${esc(x.id)}</b> — ${esc(x.snip)}</li>`).join('');
}
fetch('/api/stats').then(r=>r.json()).then(d=>{
  document.getElementById('stats').textContent = d.docs + ' documents indexed';
});
</script></body></html>"""


def make_server(rag_obj, host="127.0.0.1", port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # ponytail: silent; wire to a logger if you run this as a service

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj).encode(), "application/json; charset=utf-8", code)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/stats":
                return self._json({"docs": len(rag_obj.doc_ids())})
            text = (q.get("q", [""])[0] or "").strip()[:2000]  # trust boundary: cap the query
            try:
                k = max(1, min(20, int(q.get("k", ["5"])[0])))  # and clamp k
            except ValueError:
                k = 5
            if not text:
                return self._json({"error": "missing q"}, 400)
            if u.path == "/api/ask":
                return self._json(rag_obj.answer(text, k=k))
            if u.path == "/api/search":
                return self._json({"hits": rag_obj.search(text, k=k)})
            self._json({"error": "not found"}, 404)

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def serve(rag_obj, host="127.0.0.1", port=8765):
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: binding {host} exposes the corpus to your network", file=sys.stderr)
    print(f"rag UI on http://{host}:{port}  (Ctrl-C to stop)")
    make_server(rag_obj, host, port).serve_forever()


# --------------------------------------------------------------------------- commands
def init(flag_path=None, preset=None, model=None, key=None):
    """Write the config file. Interactive when a preset isn't given and stdin is a tty."""
    if not preset and sys.stdin.isatty():
        print("Which model backend?\n")
        for i, name in enumerate(sorted(PRESETS), 1):
            p = PRESETS[name]
            print(f"  {i}. {name:<11} {p['llm_base']}  {p.get('llm_model') or '(endpoint default)'}")
        choice = input("\nnumber [1]: ").strip() or "1"
        try:
            preset = sorted(PRESETS)[int(choice) - 1]
        except (ValueError, IndexError):
            print("not a menu number — using 'local'")
            preset = "local"
    preset = preset or "local"
    if preset not in PRESETS:
        print(f"unknown preset {preset!r}; choose from {', '.join(sorted(PRESETS))}")
        return 2
    cfg = dict(PRESETS[preset])
    if model:
        cfg["llm_model"] = model
    if key:
        cfg["llm_api_key"] = key
    path = config_path(flag_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    if key:
        os.chmod(path, 0o600)  # a key on disk shouldn't be world-readable
    print(f"wrote {path} (preset: {preset})")
    print('next: rag add <folder>   then   rag ask "your question"')
    return 0


def doctor(flag_path=None, db_flag=None):
    """Report resolved settings and probe the endpoint. Never a traceback."""
    base = resolve("llm_base", None, "RAG_LLM_BASE", "http://127.0.0.1:8888")
    key = resolve("llm_api_key", None, "RAG_API_KEY", "")
    db = resolve("db", db_flag, "RAG_DB", "rag.db")
    print(f"config:     {config_path(flag_path)}")
    print(f"llm_base:   {base}")
    print(f"llm_model:  {resolve('llm_model', None, 'RAG_LLM_MODEL', '(endpoint default)')}")
    print(f"api key:    {'set (' + key[:4] + '…)' if key else '(none)'}")
    print(f"reasoning:  {resolve('llm_reasoning', None, 'RAG_LLM_REASONING', 'auto')}")
    print(f"db:         {db}")
    try:
        with urllib.request.urlopen(_api_base(base) + "/models", timeout=5) as r:
            models = [m.get("id") for m in json.load(r).get("data", [])]
    except Exception as e:
        print(f"endpoint:   unreachable ({e})")
        print("            fix: rag init, or set llm_base in the config file")
        return 1
    print(f"endpoint:   reachable — serves {models or '(none listed)'}")
    # module name != package name for python-docx; the message must name the install
    for ext, mod, pip in ((".pdf", "pypdf", "pypdf"), (".docx", "docx", "python-docx")):
        try:
            __import__(mod)
            print(f"{ext}:       {mod} available")
        except ImportError:
            print(f"{ext}:       needs {pip} — run: uv add {pip}")
    # ffmpeg is a system binary (no pip) — it powers video frame extraction
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        print("ffmpeg:     available (needed for video)")
    except (OSError, subprocess.TimeoutExpired):
        print("ffmpeg:     not found (needed for video) — "
              "install it (apt install ffmpeg / brew install ffmpeg)")
    print("images:     read via LLM vision; a text-only model is skipped, never guessed")
    if os.path.exists(db):
        print(f"corpus:     {len(Rag(db=db).doc_ids())} documents in {db}")
    else:
        print(f"corpus:     {db} not created yet — run: rag add <folder>")
    return 0


def stats(rag_obj, flag_path=None):
    docs = rag_obj.doc_ids()
    return {
        "docs": len(docs),
        "chars": sum(len(rag_obj.doc(d)) for d in docs),
        "vectors": rag_obj._q("SELECT count(*) FROM vectors")[0][0],
        "hot": len(rag_obj.hot_tier()),
        "db": rag_obj.db_path,
        "config": config_path(flag_path),
    }


def write_glossary(flag_path=None):
    """Write the example template, and the configured glossary if it's missing."""
    body = (
        "# Domain terms that must never be rewritten by the LLM. One per line.\n"
        "# The rewriter is told to preserve these exactly (recipes.md recipe 2).\n"
        "Atlas: our internal data processing framework\n"
        "Mercury: our messaging system\n"
    )
    for path in dict.fromkeys(["glossary.example.txt", resolve("glossary", None, "RAG_GLOSSARY", "glossary.txt")]):
        if os.path.exists(path) and path != "glossary.example.txt":
            print(f"{path} already exists — leaving it alone")
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"wrote {path}")
    return 0


def _print_hits(hits):
    for h in hits:
        extra = f" [{h['sub_query']}]" if "sub_query" in h else ""
        score = h.get("sem", h.get("score", 0.0))
        print(f"{score:+.4f}  {h['id']}{extra}  {(h.get('snip') or h['text'][:70]).replace(chr(10), ' ')}")


def _print_answer(out):
    if out["answer"]:
        print(out["answer"])
    elif out["note"]:
        print(f"({out['note']})")
    for c in out["citations"]:
        print(f"  [{c['n']}] {c['id']}  {c['snip']}")


def _main(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="rag-ladder", description="6 RAG recipes, ladder-first")
    p.add_argument("--config", default=None, help="config file (default rag.json, env RAG_CONFIG)")
    p.add_argument("--db", default=None, help="SQLite file (default rag.db, env RAG_DB)")
    p.add_argument("--corpus", default=None, help="root folder for document ids (default .)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--no-llm", action="store_true", help="force deterministic rewriting")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="choose your model backend, write the config file")
    i.add_argument("--preset", default=None, choices=sorted(PRESETS))
    i.add_argument("--model", default=None, help="override the preset's model")
    i.add_argument("--key", default=None, help="API key (stored in the config file, mode 0600)")

    sub.add_parser("doctor", help="check config, endpoint, extractors, corpus")

    for name in ("add", "index"):  # `index` kept as an alias: old README muscle memory
        a = sub.add_parser(name, help="index files or folders (recursive; images via "
                                     "LLM vision, video via ffmpeg frames)")
        a.add_argument("paths", nargs="+")

    for name in ("search", "hybrid", "multi", "hot"):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("-k", type=int, default=5)

    ask = sub.add_parser("ask", help="cited answer from your documents")
    ask.add_argument("query")
    ask.add_argument("-k", type=int, default=6)
    ask.add_argument("--mode", default=None, choices=["general", "legal"],
                     help="answer prompt; legal quotes clauses verbatim and never infers")

    pre = sub.add_parser("pre", help="pre-embed (recipe 6); with a query, search stored vectors")
    pre.add_argument("query", nargs="?")
    pre.add_argument("-k", type=int, default=5)

    sv = sub.add_parser("serve", help="browser UI on a local port")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)

    sub.add_parser("stats", help="corpus and index sizes")
    sub.add_parser("glossary", help="write the glossary template")

    r = sub.add_parser("recommend", help="run the decision tree")
    r.add_argument("--qpd", type=int, default=0)
    r.add_argument("--churn", type=float, default=0.0, help="%% of docs changed per day")
    r.add_argument("--corpus-size", type=int, default=0)
    r.add_argument("--complaint", default="")
    r.add_argument("--ml", action="store_true")
    r.add_argument("--hot-patterns", action="store_true")
    r.add_argument("--no-search", action="store_true")
    r.add_argument("--latency-ok", action=argparse.BooleanOptionalAction, default=True)

    args = p.parse_args(argv)
    CFG.clear()
    CFG.update(load_config(args.config))

    if args.cmd == "init":
        return init(args.config, args.preset, args.model, args.key)
    if args.cmd == "doctor":
        return doctor(args.config, args.db)
    if args.cmd == "glossary":
        return write_glossary(args.config)
    if args.cmd == "recommend":
        rec = recommend(
            has_search=not args.no_search,
            complaint=args.complaint,
            qpd=args.qpd,
            churn_pct_day=args.churn,
            corpus=args.corpus_size,
            ml_team=args.ml,
            hot_patterns=args.hot_patterns,
            latency_ok=args.latency_ok,
        )
        print(f"recipe {rec['recipe']}: {rec['why']}")
        return 0

    db = resolve("db", args.db, "RAG_DB", "rag.db")
    corpus = resolve("corpus", args.corpus, "RAG_CORPUS", ".")
    r_obj = Rag(db=db, use_llm=not args.no_llm)

    if args.cmd in ("add", "index"):
        added = skipped = failed = 0
        for path in walk(args.paths, corpus):
            try:
                text = extract(path)
            except ExtractError as e:
                print(f"skip: {e}", file=sys.stderr)
                failed += 1
                continue
            if r_obj.add(doc_id(path, corpus), text, mtime=os.path.getmtime(path)):
                added += 1
            else:
                skipped += 1
        print(f"added {added}, skipped {skipped} (unchanged), failed {failed} -> {db}")
        return 0

    if args.cmd == "stats":
        s = stats(r_obj, args.config)
        print(json.dumps(s) if args.json else "\n".join(f"{k}: {v}" for k, v in s.items()))
        return 0

    if args.cmd == "serve":
        host = resolve("serve_host", args.host, "RAG_SERVE_HOST", "127.0.0.1")
        port = int(resolve("serve_port", args.port, "RAG_SERVE_PORT", 8765))
        return serve(r_obj, host, port)

    if args.cmd == "pre" and not args.query:
        n = r_obj.preembed()
        note = " (corpus is empty — run: rag add <path>)" if not n else ""
        print(f"pre-embedded {n} documents{note}")
        return 0

    if not r_obj.doc_ids():
        print("corpus is empty — run: rag add <path>", file=sys.stderr)
        return 3

    q = args.query
    if args.cmd == "ask":
        out = r_obj.answer(q, k=args.k, mode=args.mode)
        if args.json:
            print(json.dumps(out))
        else:
            _print_answer(out)
        return 0

    hits = []
    if args.cmd == "search":
        hits = r_obj.search(q, k=args.k)
    elif args.cmd == "hybrid":
        hits = r_obj.hybrid(q, k=args.k)
    elif args.cmd == "multi":
        hits = r_obj.search_multi_intent(q, k=args.k)
    elif args.cmd == "hot":
        r_obj.refresh_hot()
        hits = r_obj.search_hot_cold(q, k=args.k)
    elif args.cmd == "pre":
        hits = r_obj.search_preembedded(q, k=args.k)
    if args.json:
        print(json.dumps(hits))
    else:
        _print_hits(hits)
    return 0


def main(argv=None):
    """One friendly line + an exit code. Deliberately not a broad `except Exception`
    — swallowing real bugs makes 3am worse, not better."""
    try:
        return _main(argv)
    except ExtractError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error, urllib.error.URLError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
