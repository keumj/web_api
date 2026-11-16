"""Open the signal dashboard, starting the local service when necessary."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import webbrowser

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def listening(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def main():
    load_dotenv(ROOT / '.env')
    host = os.getenv('KEUMJM_HOST', '0.0.0.0')
    host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(host, host)
    port = int(os.getenv('PORT') or os.getenv('KEUMJM_PORT') or '8515')
    scheme = 'https' if os.getenv('KEUMJM_SSL_CERTFILE', '').strip() else 'http'
    url_host = f'[{host}]' if ':' in host else host
    url = f'{scheme}://{url_host}:{port}/global-signals'
    if not listening(host, port):
        log_path = ROOT / 'outputs' / 'signals-service.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'PYTHONUTF8': '1'}
        with log_path.open('ab') as log:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / 'scripts' / 'run_uvicorn.py')],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                start_new_session=os.name != 'nt',
            )
        print('Starting local service...', flush=True)
        deadline = time.monotonic() + 60
        while not listening(host, port):
            if process.poll() is not None or time.monotonic() >= deadline:
                print(f'Service is not ready. Check {log_path}')
                return 1
            time.sleep(.5)
    print(f'Opening {url}', flush=True)
    if not webbrowser.open(url, new=2):
        print('Browser could not be opened automatically. Open the URL above manually.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
