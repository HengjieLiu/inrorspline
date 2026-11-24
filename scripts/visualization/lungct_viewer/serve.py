"""Read-only, loopback-only by default; no raw dataset or arbitrary path access."""
from __future__ import annotations

import argparse
from functools import partial
import http.server
import json
import mimetypes
from pathlib import Path
import re
import signal
import socketserver
from urllib.parse import unquote, urlsplit


class Handler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.respond(False)

    def do_GET(self):
        self.respond(True)

    def respond(self, body):
        name=unquote(urlsplit(self.path).path)
        if name=='/api/catalog':
            root=self.server.bundle_root; relative='catalog.json'
        elif name.startswith('/assets/'):
            root=self.server.bundle_root; relative=name[len('/assets/'):]
            if not re.fullmatch(r'cases/(?:4dct|copd)_\d{2}/(?:manifest\.json|[a-z0-9_]+\.bin\.gz)',relative):
                self.send_error(404);return
        elif name.startswith('/vendor/'):
            root=self.server.bundle_root/'deps';relative=name[len('/vendor/'):]
            if relative not in ('three.module.js','three.core.js','OrbitControls.js','LICENSE'):
                self.send_error(404);return
        else:
            root=self.server.web_root;relative=name.lstrip('/') or 'index.html'
            if name=='/' and getattr(self.server,'viewer','standard')=='grid-compare':
                relative='grid_compare/index.html'
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            self.send_error(404);return
        # Public case manifests omit private operational source paths.
        if path.name=='manifest.json' and root==self.server.bundle_root:
            obj=json.loads(path.read_text());obj.pop('source_hashes',None)
            payload=json.dumps(obj,allow_nan=False).encode()
        else:payload=None
        etag=f'"{path.stat().st_mtime_ns:x}-{path.stat().st_size:x}"'
        if self.headers.get('If-None-Match')==etag:
            self.send_response(304);self.end_headers();return
        self.send_response(200)
        self.send_header('Content-Type','application/octet-stream' if path.name.endswith('.bin.gz') else mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(len(payload) if payload is not None else path.stat().st_size))
        if path.name.endswith('.bin.gz'):self.send_header('Content-Encoding','gzip')
        self.send_header('Cache-Control','private, max-age=3600' if path.name.endswith('.bin.gz') else 'no-cache')
        self.send_header('ETag',etag)
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.end_headers()
        if not body:return
        try:
            if payload is not None:self.wfile.write(payload)
            else:
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(1024*1024),b''):self.wfile.write(block)
        except (BrokenPipeError,ConnectionResetError):pass


class Server(socketserver.ThreadingMixIn,http.server.HTTPServer):
    daemon_threads=True
    allow_reuse_address=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root',type=Path,required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=1101)
    p.add_argument('--viewer',choices=('standard','grid-compare'),default='standard')
    a=p.parse_args()
    def terminate(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,terminate)
    if not (a.bundle_root/'catalog.json').is_file():p.error('No prepared catalog')
    with Server((a.host,a.port),Handler) as server:
        server.bundle_root=a.bundle_root.resolve();server.web_root=Path(__file__).parent/'web'
        server.viewer=a.viewer
        print(f'DIR-Lab viewer http://{a.host}:{a.port}',flush=True)
        server.serve_forever()


if __name__=='__main__':main()
