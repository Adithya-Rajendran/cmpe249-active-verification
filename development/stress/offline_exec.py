"""Run a Python script with outbound IP sockets disabled before imports."""
from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import runpy
import socket
import sys

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_sendto = socket.socket.sendto
_original_sendmsg = getattr(socket.socket, 'sendmsg', None)


def _guard(original):
    def call(self, *args, **kwargs):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise PermissionError('outbound IP sockets are disabled for this run')
        return original(self, *args, **kwargs)
    return call


socket.socket.connect = _guard(_original_connect)
socket.socket.connect_ex = _guard(_original_connect_ex)
socket.socket.sendto = _guard(_original_sendto)
if _original_sendmsg:
    socket.socket.sendmsg = _guard(_original_sendmsg)


def _no_resolution(*args, **kwargs):
    raise PermissionError('network name resolution is disabled for this run')


socket.getaddrinfo = _no_resolution
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', YOLO_AUTOINSTALL='false')
if len(sys.argv) == 2 and sys.argv[1] == '--probe':
    for family, destination in ((socket.AF_INET, ('127.0.0.1', 9)), (socket.AF_INET6, ('::1', 9))):
        try:
            with socket.socket(family) as s:
                s.connect(destination)
        except PermissionError:
            continue
        raise SystemExit('socket guard failed')
    print('PASS IPv4 and IPv6 connections denied before script imports')
else:
    if len(sys.argv) < 2:
        raise SystemExit('usage: offline_exec.py SCRIPT [ARGS...]')
    if len(sys.argv) < 4 or sys.argv[1] != '--guard-record':
        raise SystemExit('usage: offline_exec.py --guard-record PATH SCRIPT [ARGS...]')
    record_path = Path(sys.argv[2])
    script = str(Path(sys.argv[3]).resolve(strict=True))
    from perception.common import configure_ultralytics
    configure_ultralytics()
    import ultralytics.nn.text_model as text_model
    clip = (text_model.WEIGHTS_DIR / 'clip/ViT-B-32.pt').resolve(strict=True)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    clip_hash = digest(clip)
    if clip_hash != '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af':
        raise ValueError('resolved CLIP checkpoint differs from frozen hash')
    source = Path(text_model.__file__).resolve(strict=True)
    with record_path.open('x') as f:
        json.dump({'clip_actual_path': str(clip), 'clip_sha256': clip_hash,
                   'text_model_source': str(source), 'text_model_sha256': digest(source),
                   'guard_sha256': digest(__file__), 'script': script, 'argv': sys.argv[4:],
                   'python_socket_policy': 'IPv4/IPv6 connect and send disabled; DNS disabled'}, f, indent=2)
        f.write('\n')
    sys.argv = [script, *sys.argv[4:]]
    runpy.run_path(script, run_name='__main__')
