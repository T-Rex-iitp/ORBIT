#!/usr/bin/env python3
"""Connect to ADS-B Raw and SBS TCP feeds at the same time.

The defaults follow the existing C++ client implementation in this repository:
- Raw feed: TCP 30002
- SBS feed: TCP 5002

Examples:
    python ADS-B-Display/python/sbs_raw_connector.py --host 127.0.0.1
    python ADS-B-Display/python/sbs_raw_connector.py --host 10.0.0.15 \
        --raw-output raw.log --sbs-output sbs.log --duration 60
"""

from __future__ import annotations

import argparse
import signal
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO


@dataclass(frozen=True)
class FeedConfig:
    name: str
    host: str
    port: int
    output_path: Optional[Path]


class FeedWorker(threading.Thread):
    def __init__(self, config: FeedConfig, stop_event: threading.Event, reconnect_delay: float) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.reconnect_delay = reconnect_delay
        self.output_fp: Optional[TextIO] = None

    def run(self) -> None:
        try:
            if self.config.output_path is not None:
                self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_fp = self.config.output_path.open("a", encoding="utf-8")

            while not self.stop_event.is_set():
                try:
                    self._stream_once()
                except (ConnectionError, OSError, socket.timeout) as exc:
                    if self.stop_event.is_set():
                        break
                    print(f"[{self.config.name}] connection issue: {exc}. reconnecting in {self.reconnect_delay:.1f}s")
                    self.stop_event.wait(self.reconnect_delay)
        finally:
            if self.output_fp is not None:
                self.output_fp.close()

    def _stream_once(self) -> None:
        print(f"[{self.config.name}] connecting to {self.config.host}:{self.config.port} ...")
        with socket.create_connection((self.config.host, self.config.port), timeout=10) as conn:
            conn.settimeout(1.0)
            print(f"[{self.config.name}] connected")
            with conn.makefile("r", encoding="utf-8", errors="replace", newline="\n") as reader:
                while not self.stop_event.is_set():
                    try:
                        line = reader.readline()
                    except socket.timeout:
                        continue

                    if not line:
                        raise ConnectionError("remote peer closed the connection")

                    line = line.rstrip("\r\n")
                    timestamp_ms = int(time.time() * 1000)
                    print(f"[{self.config.name}] {line}")
                    if self.output_fp is not None:
                        self.output_fp.write(f"{timestamp_ms}\n{line}\n")
                        self.output_fp.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual connector for SBS and Raw ADS-B feeds")
    parser.add_argument("--host", default="127.0.0.1", help="Feed server host (default: %(default)s)")
    parser.add_argument("--raw-port", type=int, default=30002, help="Raw feed TCP port (default: %(default)s)")
    parser.add_argument("--sbs-port", type=int, default=5002, help="SBS feed TCP port (default: %(default)s)")
    parser.add_argument("--raw-output", type=Path, help="Optional file for recorded raw stream")
    parser.add_argument("--sbs-output", type=Path, help="Optional file for recorded SBS stream")
    parser.add_argument("--duration", type=float, default=0.0, help="Auto-stop after N seconds (0 = run forever)")
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before reconnecting after failures (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()

    def handle_signal(_sig: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    workers = [
        FeedWorker(
            FeedConfig(
                name="RAW",
                host=args.host,
                port=args.raw_port,
                output_path=args.raw_output,
            ),
            stop_event=stop_event,
            reconnect_delay=args.reconnect_delay,
        ),
        FeedWorker(
            FeedConfig(
                name="SBS",
                host=args.host,
                port=args.sbs_port,
                output_path=args.sbs_output,
            ),
            stop_event=stop_event,
            reconnect_delay=args.reconnect_delay,
        ),
    ]

    for worker in workers:
        worker.start()

    if args.duration > 0:
        stop_event.wait(args.duration)
        stop_event.set()
    else:
        while not stop_event.is_set():
            time.sleep(0.2)

    for worker in workers:
        worker.join()

    print("Stopped SBS/Raw connectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
