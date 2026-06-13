#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
from pathlib import Path


PATTERNS = [
    ("time_elapsed_sec", re.compile(r"Time elapsed \(FINAL\)\s*:\s*([0-9.]+)\s+secs")),
    ("throughput_req_per_sec", re.compile(r"Throughput \(Req/sec\)\s*:\s*([0-9.]+)")),
    ("avg_latency_ms", re.compile(r"Avg latency .*?\(ms\)\s*:\s*([0-9.]+)")),
    ("avg_ev_lookup_ms", re.compile(r"Avg ev lookup time .*?\(ms\)\s*:\s*([0-9.]+)")),
    ("avg_apply_emb_ms", re.compile(r"Avg apply emb time .*?\(ms\)\s*:\s*([0-9.]+)")),
    ("cpu_cal_ms", re.compile(r"CPU cal time \(ms\)\s*:\s*([0-9.]+)")),
    ("pim_ev_trans_ms", re.compile(r"\(PIM \+ EV_trans\) time \(ms\)\s*:\s*([0-9.]+)")),
    ("ev_transmission_ms", re.compile(r"EV transmission time \(ms\)\s*:\s*([0-9.]+)")),
    ("network_gpu_ms", re.compile(r"\(network \+ GPU\) time \(ms\)\s*:\s*([0-9.]+)")),
    ("others_ms", re.compile(r"others time \(ms\)\s*:\s*([0-9.]+)")),
    ("num_latency_samples", re.compile(r"len\s*:\s*([0-9]+)")),
]


def parse_file(path: Path) -> dict:
    metrics = {"file": str(path)}
    text = path.read_text(errors="replace")

    for line in text.splitlines():
        for key, pattern in PATTERNS:
            match = pattern.search(line)
            if match:
                value = match.group(1)
                metrics[key] = int(value) if key == "num_latency_samples" else float(value)

    return metrics


def write_csv(rows: list[dict], stream) -> None:
    keys = ["file"]
    for key, _ in PATTERNS:
        keys.append(key)

    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse CloverRec benchmark logs")
    parser.add_argument("logs", nargs="+", type=Path, help="Log file(s) to parse")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    args = parser.parse_args()

    rows = [parse_file(path) for path in args.logs]

    if args.format == "json":
        json.dump(rows, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        write_csv(rows, sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
