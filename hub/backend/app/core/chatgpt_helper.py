"""Legion SIWC loopback helper. Python 3.10+, standard library only.
Run on the computer with your browser, including for a remote/container Hub.
This file never receives or stores OpenAI access/refresh tokens.
"""
import argparse
import hmac
import http.server
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

# Filled by the Hub download action; this is a one-use, ten-minute setup capability.
LEGION_SETUP = None

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

def validate_hub(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("Use the Hub origin, without credentials, paths, or query strings")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")):
        raise ValueError("Remote Hubs require HTTPS")
    return url.rstrip("/")

def main():
    parser = argparse.ArgumentParser(description="Connect your ChatGPT plan to your Legion Hub")
    parser.add_argument("--hub")
    parser.add_argument("--ticket")
    args = parser.parse_args()
    setup = LEGION_SETUP or {}
    hub = validate_hub(args.hub or setup.get("hub") or input("Hub URL: ").strip())
    ticket = args.ticket or setup.get("ticket") or input("Setup ticket: ").strip()
    opener = urllib.request.build_opener(NoRedirect())
    def post(route, body):
        request = urllib.request.Request(hub + "/api/v1/providers/chatgpt" + route,
            data=json.dumps({"ticket": ticket, **body}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with opener.open(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.load(error).get("detail", "Connection failed")
            except (ValueError, AttributeError):
                detail = "Connection failed"
            raise RuntimeError(str(detail)) from None

    done = False
    state = None
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Authorization codes/state must never enter console or HTTP logs.
        def do_GET(self):
            nonlocal done
            parsed = urllib.parse.urlsplit(self.path)
            values = urllib.parse.parse_qs(parsed.query)
            expected_host = f"127.0.0.1:{self.server.server_port}"
            if (self.headers.get("Host") != expected_host or parsed.path != "/auth/callback"
                or len(values.get("state", [])) != 1
                or not hmac.compare_digest(values["state"][0], state or "")):
                self.send_error(400, "Invalid callback")
                return
            try:
                result = post("/helper/complete", {key: values[key][0] for key in ("state", "code", "client_id", "error") if key in values})
                if not result.get("connected"):
                    raise RuntimeError(result.get("error") or "ChatGPT plan usage was not authorized")
                message = "ChatGPT connected. Return to Legion and close this tab."
            except RuntimeError as error:
                print(str(error))
                message = "Connection failed. Return to Legion and start again."
            done = True
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(message.encode())
    with http.server.HTTPServer(("127.0.0.1", 0), Handler) as server:
        server.timeout = 1
        server.socket.settimeout(10)
        result = post("/helper/start", {"port": server.server_port})
        state = result["state"]
        url = result["authorize_url"]
        if not url.startswith("https://auth.openai.com/api/accounts/authorize?"):
            raise ValueError("Unexpected authorization endpoint")
        print("Opening ChatGPT authorization in your browser. Keep this window open.")
        if not webbrowser.open(url):
            print("Open this authorization URL in a browser on THIS computer:")
            print(url)
        deadline = time.monotonic() + 600
        while not done and time.monotonic() < deadline:
            server.handle_request()
        if not done:
            raise RuntimeError("Setup expired. Start again in Legion Settings.")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(str(error))
        raise SystemExit(1)
