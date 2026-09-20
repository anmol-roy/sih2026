"""
IP-SAKTI server launcher.
Loads .env, sets PYTHONPATH, then starts uvicorn programmatically
so the env vars are guaranteed to be in scope before any imports.

Run:  .venv\Scripts\python.exe run_server.py
"""
import os
import socket
import sys
from pathlib import Path

# ── 1. Load .env ────────────────────────────────────────────────────────────
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)
        print(f"  env: {key} loaded ({len(val)} chars)")
else:
    print("WARNING: .env not found")

# ── 2. Add src/ to path so bare imports work ────────────────────────────────
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# ── 3. Prefer the project's virtual environment when available ───────────────
venv_python = Path(__file__).parent / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if venv_python.exists() and str(sys.executable).lower() != str(venv_python).lower():
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__)), *sys.argv[1:]])

# ── 4. Start uvicorn ────────────────────────────────────────────────────────
import uvicorn  # noqa: E402


def _pick_port(start_port: int = 8000, max_tries: int = 20) -> int:
    preferred = int(os.getenv("PORT", str(start_port)))
    for port in range(preferred, preferred + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free port found in range {preferred}-{preferred + max_tries - 1}. "
        "Stop the process using the port or set PORT to a free value."
    )


if __name__ == "__main__":
    port = _pick_port()
    os.environ["PORT"] = str(port)
    print(f"\nStarting IP-SAKTI on http://127.0.0.1:{port}\n")
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        log_level="info",
    )
