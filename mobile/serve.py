"""
Pokreće RehabPose demo i pravi HTTPS link za telefon (S25 Ultra).

Na računaru (Chrome):  http://127.0.0.1:8765
Na telefonu:          javni HTTPS URL koji se ispiše (Cloudflare tunnel).

Kamera u pregledaču radi samo na localhost ili HTTPS.
"""
from __future__ import annotations

import http.server
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8765
HTTPS_PORT = 8766
CLOUDFLARED = ROOT / "tools" / "cloudflared.exe"


def write_png(path: Path, size: int) -> None:
    """Jednostavna teal ikona sa belim skeletnim likom."""

    def px(x: int, y: int) -> tuple[int, int, int, int]:
        nx = (x + 0.5) / size
        ny = (y + 0.5) / size
        cx, cy = 0.5, 0.5
        dx, dy = nx - cx, ny - cy
        r = (dx * dx + dy * dy) ** 0.5
        if r > 0.46:
            return (0, 0, 0, 0)
        if r > 0.40:
            return (14, 35, 44, 255)
        bg = (16, 122, 104, 255)
        def near(px_, py_, rad):
            return (nx - px_) ** 2 + (ny - py_) ** 2 <= rad * rad
        def on_seg(x0, y0, x1, y1, w=0.028):
            vx, vy = x1 - x0, y1 - y0
            lx = nx - x0
            ly = ny - y0
            den = vx * vx + vy * vy
            t = 0 if den == 0 else max(0, min(1, (lx * vx + ly * vy) / den))
            qx, qy = x0 + t * vx, y0 + t * vy
            return (nx - qx) ** 2 + (ny - qy) ** 2 <= w * w
        white = (243, 247, 248, 255)
        if near(0.50, 0.28, 0.055):
            return white
        if on_seg(0.50, 0.34, 0.50, 0.55) or on_seg(0.50, 0.40, 0.34, 0.50) or on_seg(0.50, 0.40, 0.66, 0.50):
            return white
        if on_seg(0.50, 0.55, 0.38, 0.74) or on_seg(0.50, 0.55, 0.62, 0.74):
            return white
        return bg

    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            raw.extend(px(x, y))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + tag + data + crc.to_bytes(4, "big")

    ihdr = size.to_bytes(4, "big") + size.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def ensure_icons() -> None:
    write_png(ROOT / "icons" / "icon-192.png", 192)
    write_png(ROOT / "icons" / "icon-512.png", 512)


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert() -> tuple[Path, Path]:
    cert = ROOT / "tools" / "cert.pem"
    key = ROOT / "tools" / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    openssl = Path(r"C:\Program Files\Git\usr\bin\openssl.exe")
    if not openssl.exists():
        raise FileNotFoundError("openssl nije pronađen")
    cert.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            str(openssl), "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-days", "14",
            "-nodes", "-keyout", str(key), "-out", str(cert),
            "-subj", "/CN=RehabPose/O=RehabPose/C=RS",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    return cert, key


def download_cloudflared() -> Path | None:
    if CLOUDFLARED.exists():
        return CLOUDFLARED
    CLOUDFLARED.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    print("Preuzimam cloudflared (HTTPS tunel za telefon)...", flush=True)
    try:
        urllib.request.urlretrieve(url, CLOUDFLARED)
        return CLOUDFLARED
    except Exception as exc:
        print(f"Cloudflare tunel nije preuzet: {exc}")
        return None


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".webmanifest": "application/manifest+json",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".png": "image/png",
        ".css": "text/css",
        ".html": "text/html",
        ".json": "application/json",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        sys.stdout.write("[http] " + (fmt % args) + "\n")

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def serve(httpd) -> None:
    httpd.serve_forever()


def start_http() -> http.server.ThreadingHTTPServer:
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), QuietHandler)
    threading.Thread(target=serve, args=(httpd,), daemon=True).start()
    return httpd


def start_https(cert: Path, key: Path) -> http.server.ThreadingHTTPServer | None:
    try:
        httpd = http.server.ThreadingHTTPServer(("0.0.0.0", HTTPS_PORT), QuietHandler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        threading.Thread(target=serve, args=(httpd,), daemon=True).start()
        return httpd
    except Exception as exc:
        print(f"HTTPS na LAN-u nije podignut: {exc}")
        return None


def start_tunnel(exe: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [str(exe), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def main() -> None:
    os.chdir(ROOT)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ensure_icons()
    httpd = start_http()
    ip = local_ip()
    print()
    print("RehabPose demo radi.", flush=True)
    print(f"  Racunar (kamera OK):  http://127.0.0.1:{PORT}/", flush=True)
    print("  Telefon: sacekajte HTTPS tunel ispod", flush=True)
    print()

    tunnel_url = None
    proc = None
    exe = download_cloudflared()
    if exe:
        proc = start_tunnel(exe)
        deadline = time.time() + 45
        print("Pokrecem Cloudflare tunel...", flush=True)
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            if "trycloudflare.com" in line:
                for part in line.split():
                    if part.startswith("https://") and "trycloudflare.com" in part:
                        tunnel_url = part.strip()
                        break
            if tunnel_url:
                break
        if tunnel_url:
            print()
            print("  TELEFON (Chrome na S25 Ultra):", flush=True)
            print(f"  {tunnel_url}", flush=True)
            print()
            print("  Na telefonu: Chrome meni -> Dodaj na pocetni ekran / Install app", flush=True)
            print("  Zatim: dozvoli kameru, nasloni telefon 2-3 m, celo telo u kadar.", flush=True)
        else:
            print("  Tunel nije dao URL. Koristite LAN HTTPS ili localhost na racunaru.", flush=True)

    print()
    print("Ctrl+C za zaustavljanje.", flush=True)
    try:
        while True:
            time.sleep(1)
            if proc and proc.poll() is not None and tunnel_url is None:
                print("cloudflared se ugasio.")
                break
    except KeyboardInterrupt:
        print("\nZaustavljam...")
    finally:
        httpd.shutdown()
        if proc and proc.poll() is not None:
            pass
        elif proc:
            proc.terminate()


if __name__ == "__main__":
    main()
