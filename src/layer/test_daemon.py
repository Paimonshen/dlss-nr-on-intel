#!/usr/bin/env python3
"""Exercise request bounds, codecs and the native layer's disconnect behavior."""
import contextlib
import io
import pathlib
import socket
import struct
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import numpy as np
import nr_daemon as daemon

ROOT = pathlib.Path(__file__).resolve().parents[2]


class Request:
    def __init__(self, header, payload=b''):
        self.header, self.payload, self.reads, self.reply = header, payload, 0, None

    def recv(self, count):
        self.reads += 1
        if self.header:
            value, self.header = self.header, b''
            return value
        value, self.payload = self.payload[:count], self.payload[count:]
        return value

    def sendall(self, data):
        self.reply = data


def request_tests():
    args = SimpleNamespace(max_pixels=1024, profile='standard', intensity=1,
                           detail_strength=1, colour_strength=1, dump=None)
    for magic, width, height in ((0, 1, 1), (daemon.MAGIC, 0, 8),
                                 (daemon.MAGIC, 8, 0), (daemon.MAGIC, 1024, 1024)):
        request = Request(struct.pack('<4I', magic, width, height, 44))
        try:
            daemon.process_connection(request, None, args)
        except ValueError:
            assert request.reads == 1 and request.reply is None
        else:
            raise AssertionError('invalid header accepted')
    raw = bytes(range(64))
    request = Request(struct.pack('<4I', daemon.MAGIC, 4, 4, 999), raw)
    with contextlib.redirect_stdout(io.StringIO()):
        daemon.process_connection(request, None, args)
    assert request.reply == raw
    for vk_format in daemon.FORMATS:
        rgb = daemon.decode(raw, 4, 4, vk_format)
        assert daemon.encode(rgb, raw, vk_format) == raw
    try:
        daemon.receive(Request(b'', b'123'), 4)
    except EOFError:
        pass
    else:
        raise AssertionError('truncated body accepted')


def native_exchange_tests():
    with tempfile.TemporaryDirectory(prefix='nr-exchange-') as temporary:
        for mode in ('echo', 'reject', 'partial'):
            path = str(pathlib.Path(temporary) / mode)
            errors = []
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(path)
                server.listen(1)
                server.settimeout(5)

                def serve():
                    try:
                        connection, _ = server.accept()
                        with connection:
                            connection.settimeout(5)
                            header = daemon.receive(connection, 16)
                            _, width, height, _ = struct.unpack('<4I', header)
                            if mode == 'reject':
                                return  # client may still be writing a large payload
                            body = daemon.receive(connection, width * height * 4)
                            connection.sendall(body if mode == 'echo' else body[:17])
                    except Exception as error:
                        errors.append(error)

                thread = threading.Thread(target=serve)
                thread.start()
                result = subprocess.run([str(ROOT / 'work' / 'test_exchange'), path,
                                         'echo' if mode == 'echo' else 'reject'],
                                        capture_output=True, timeout=10)
                thread.join(timeout=6)
                assert not thread.is_alive() and not errors, errors
                assert result.returncode == 0, (mode, result.returncode, result.stderr)
    print('daemon/layer: early bounds, HDR/alpha round trips, truncation and SIGPIPE-safe exchange OK')


if __name__ == '__main__':
    request_tests()
    native_exchange_tests()
