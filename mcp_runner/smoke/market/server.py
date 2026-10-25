import json
from http.server import BaseHTTPRequestHandler, HTTPServer

ITEM = {
    "type": "mcp",
    "is_active": True,
    "slug": "managed-fixture",
    "isolation": "tenant_aware",
    "internal_endpoint": "http://runner:8080/mcp",
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib handler API
        if self.path == "/api/internal/gateway/items/slug/managed-fixture":
            body = json.dumps(ITEM).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args):
        return


HTTPServer(("0.0.0.0", 8300), Handler).serve_forever()
