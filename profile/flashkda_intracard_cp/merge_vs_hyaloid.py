# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Merge cuLA bench_vs_hyaloid CSV with hyaloid bench_intracard_cp_sm90.py log."""
import csv
import math
import os
import re

CULA_CSV = os.path.join(os.path.dirname(__file__), "benchmark_vs_hyaloid.csv")
HYALOID_LOG = "/tmp/bench_hyaloid_sm90.log"
OUT_CSV = os.path.join(os.path.dirname(__file__), "benchmark_vs_hyaloid_merged.csv")

ROW_RE = re.compile(
    r"^\s*(\S.*?)\s+(\d+)\s+([YN])\s+(\d+)\s+│\s+\S+\s+\S+\s+│"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x"
)


def parse_hyaloid(path):
    rows = {}
    h = None
    with open(path) as f:
        for line in f:
            mh = re.match(r"\s*\[H=(\d+)\]", line)
            if mh:
                h = int(mh.group(1))
                continue
            m = ROW_RE.match(line)
            if m and h is not None:
                tag = m.group(1).strip()
                pred = m.group(3)
                rows[(h, tag)] = {
                    "pred": pred,
                    "n_sub": int(m.group(4)),
                    "off_ms": float(m.group(5)),
                    "on_ms": float(m.group(6)),
                    "speedup": float(m.group(7)),
                }
    return rows


def parse_cula(path):
    rows = {}
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows[(int(r["H"]), r["config"])] = r
    return rows


def main():
    hy = parse_hyaloid(HYALOID_LOG)
    cu = parse_cula(CULA_CSV)
    common = sorted(set(hy) & set(cu), key=lambda x: (x[0], list(cu).index(x)))
    cula_wins = []
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["H", "config", "T",
                     "hy_serial_ms", "hy_cp_on_ms", "hy_cp_speedup", "hy_pred", "hy_n_sub",
                     "cula_serial_ms", "cula_cp_ms", "cula_cp_speedup",
                     "cula_vs_hy_cp_x", "cula_vs_hy_serial_x"])
        print(f"  {'H':>1s}  {'config':<24s}  {'T':>7s}  "
              f"{'hy_off':>7s}  {'hy_on':>7s} hyP  "
              f"{'cu_ser':>7s}  {'cu_cp':>7s}  "
              f"{'cu/hy CP':>9s}  {'cu/hy ser':>9s}")
        for h, tag in common:
            h_row = hy[(h, tag)]
            c_row = cu[(h, tag)]
            cu_cp = float(c_row["cula_cp_auto_ms"])
            cu_se = float(c_row["cula_serial_ms"])
            r_cp = h_row["on_ms"] / cu_cp
            r_se = h_row["off_ms"] / cu_se
            T = int(c_row["total_T"])
            wr.writerow([h, tag, T,
                         f"{h_row['off_ms']:.4f}", f"{h_row['on_ms']:.4f}",
                         f"{h_row['speedup']:.2f}", h_row["pred"], h_row["n_sub"],
                         f"{cu_se:.4f}", f"{cu_cp:.4f}", c_row["cula_cp_speedup"],
                         f"{r_cp:.2f}", f"{r_se:.2f}"])
            cula_wins.append(r_cp)
            print(f"  {h:>1d}  {tag:<24s}  {T:>7d}  "
                  f"{h_row['off_ms']:>7.3f}  {h_row['on_ms']:>7.3f}  {h_row['pred']}   "
                  f"{cu_se:>7.3f}  {cu_cp:>7.3f}  "
                  f"{r_cp:>8.2f}x  {r_se:>8.2f}x")

    geo = math.exp(sum(math.log(x) for x in cula_wins) / len(cula_wins))
    fastest = max(cula_wins)
    slowest = min(cula_wins)
    losses = [x for x in cula_wins if x < 1.0]
    wins = [x for x in cula_wins if x >= 1.0]
    print(f"\n  CP A/B  cula/hyaloid: geo-mean={geo:.2f}x  best={fastest:.2f}x  worst={slowest:.2f}x")
    print(f"  cula wins: {len(wins)}/{len(cula_wins)}  losses: {len(losses)} (configs: "
          + ", ".join(f"{h}:{t}" for (h, t), x in zip(common, cula_wins) if x < 1.0) + ")")
    print(f"\n  wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
