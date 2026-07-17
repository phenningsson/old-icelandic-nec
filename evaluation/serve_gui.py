"""Serve the cluster-explorer GUIs over http://localhost:8000/ .

Run once (it keeps running); then just refresh the browser tab after re-rendering
with `evaluation.tune` or `evaluation.visualize_clusters`:

    python -m evaluation.serve_gui            # http://localhost:8000/

Serves evaluation/out/gui/ with no-cache headers (so a refresh always shows the
latest render) and an auto index that links every *_cluster_explorer.html found.
"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
GUI_DIR = _REPO / "evaluation" / "out" / "gui"


class _Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._index()
        return super().do_GET()

    def _index(self):
        root = Path(self.directory)
        links = sorted(root.rglob("*_cluster_explorer.html"))
        items = "".join(
            f'<li><a href="/{p.relative_to(root)}">{p.relative_to(root)}</a></li>'
            for p in links) or "<li>(no renders yet — run evaluation.tune)</li>"
        body = (f"<!doctype html><meta charset=utf-8><title>Cluster explorers</title>"
                f"<h1>Cluster explorers</h1><ul>{items}</ul>").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dir", default=str(GUI_DIR))
    args = ap.parse_args(argv)
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    handler = partial(_Handler, directory=args.dir)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {args.dir} at http://localhost:{args.port}/  (Ctrl-C to stop)")
    httpd.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
