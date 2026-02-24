#!/usr/bin/env python3
"""
nsys profile analyzer for legate_0.nsys-rep
Extracts performance bottlenecks from the SQLite export.

Usage:
    nsys export --type sqlite --output /tmp/legate_0.sqlite legate_0.nsys-rep
    python3 analyze_profile.py [/tmp/legate_0.sqlite]
"""

import sqlite3
import sys
import os
from collections import defaultdict

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/legate_0.sqlite"
TOP_N = 20

SEP = "=" * 80

def fmt_ns(ns):
    """Format nanoseconds to human-readable."""
    if ns is None:
        return "N/A"
    if ns >= 1e9:
        return f"{ns/1e9:.3f}s"
    if ns >= 1e6:
        return f"{ns/1e6:.3f}ms"
    if ns >= 1e3:
        return f"{ns/1e3:.3f}us"
    return f"{ns:.0f}ns"

def fmt_bytes(b):
    if b is None:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}TB"

def pct(part, total):
    if total == 0:
        return 0.0
    return 100.0 * part / total

def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def run():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ------------------------------------------------------------------ #
    # 0. Overall time range
    # ------------------------------------------------------------------ #
    header("0. OVERALL TIMELINE")
    cur.execute("""
        SELECT MIN(start) as t0, MAX(end) as t1
        FROM (
            SELECT MIN(start) as start, MAX(end) as end FROM CUPTI_ACTIVITY_KIND_KERNEL
            UNION ALL
            SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_MEMCPY
        )
    """)
    row = cur.fetchone()
    t0, t1 = row["t0"], row["t1"]
    total_wall = t1 - t0
    print(f"  Total wall time : {fmt_ns(total_wall)}")
    print(f"  Start (ns)      : {t0}")
    print(f"  End   (ns)      : {t1}")

    # ------------------------------------------------------------------ #
    # 1. GPU kernel time breakdown — top kernels by total duration
    # ------------------------------------------------------------------ #
    header("1. TOP GPU KERNELS BY TOTAL TIME")
    cur.execute("""
        SELECT
            s.value AS kernel_name,
            COUNT(*) AS calls,
            SUM(end - start) AS total_ns,
            AVG(end - start) AS avg_ns,
            MIN(end - start) AS min_ns,
            MAX(end - start) AS max_ns,
            SUM(gridX * gridY * gridZ) AS total_blocks
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        GROUP BY k.shortName
        ORDER BY total_ns DESC
        LIMIT ?
    """, (TOP_N,))
    rows = cur.fetchall()
    cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    total_kernel_ns = cur.fetchone()[0] or 1
    print(f"  {'Kernel':<55} {'Calls':>7} {'Total':>10} {'%Total':>7} {'Avg':>10} {'Max':>10}")
    print(f"  {'-'*55} {'-'*7} {'-'*10} {'-'*7} {'-'*10} {'-'*10}")
    for r in rows:
        name = r["kernel_name"][:54]
        print(f"  {name:<55} {r['calls']:>7} {fmt_ns(r['total_ns']):>10} "
              f"{pct(r['total_ns'], total_kernel_ns):>6.1f}% "
              f"{fmt_ns(r['avg_ns']):>10} {fmt_ns(r['max_ns']):>10}")

    # ------------------------------------------------------------------ #
    # 2. Memory copy analysis
    # ------------------------------------------------------------------ #
    header("2. MEMORY COPY ANALYSIS")
    cur.execute("""
        SELECT
            e1.label AS copy_kind,
            e2.label AS src_kind,
            e3.label AS dst_kind,
            COUNT(*) AS count,
            SUM(bytes) AS total_bytes,
            SUM(end - start) AS total_ns,
            AVG(bytes) AS avg_bytes,
            MAX(bytes) AS max_bytes
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        LEFT JOIN ENUM_CUDA_MEMCPY_OPER e1 ON e1.id = m.copyKind
        LEFT JOIN ENUM_CUDA_MEM_KIND e2 ON e2.id = m.srcKind
        LEFT JOIN ENUM_CUDA_MEM_KIND e3 ON e3.id = m.dstKind
        GROUP BY m.copyKind, m.srcKind, m.dstKind
        ORDER BY total_ns DESC
    """)
    rows = cur.fetchall()
    cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    total_memcpy_ns = cur.fetchone()[0] or 1
    cur.execute("SELECT SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    total_memcpy_bytes = cur.fetchone()[0] or 1

    print(f"  Total memcpy time  : {fmt_ns(total_memcpy_ns)}")
    print(f"  Total data moved   : {fmt_bytes(total_memcpy_bytes)}")
    print()
    print(f"  {'Kind':<35} {'Count':>7} {'Bytes':>10} {'Time':>10} {'%Time':>7} {'BW GB/s':>9}")
    print(f"  {'-'*35} {'-'*7} {'-'*10} {'-'*10} {'-'*7} {'-'*9}")
    for r in rows:
        kind = f"{r['copy_kind']} {r['src_kind']}→{r['dst_kind']}"[:34]
        bw = (r["total_bytes"] / r["total_ns"]) if r["total_ns"] > 0 else 0  # GB/s (ns→s cancel)
        print(f"  {kind:<35} {r['count']:>7} {fmt_bytes(r['total_bytes']):>10} "
              f"{fmt_ns(r['total_ns']):>10} {pct(r['total_ns'], total_memcpy_ns):>6.1f}% {bw:>9.1f}")

    # ------------------------------------------------------------------ #
    # 3. Synchronization overhead
    # ------------------------------------------------------------------ #
    header("3. SYNCHRONIZATION OVERHEAD")
    cur.execute("""
        SELECT
            e.label AS sync_type,
            COUNT(*) AS count,
            SUM(end - start) AS total_ns,
            AVG(end - start) AS avg_ns,
            MAX(end - start) AS max_ns
        FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION s
        LEFT JOIN ENUM_CUPTI_SYNC_TYPE e ON e.id = s.syncType
        GROUP BY s.syncType
        ORDER BY total_ns DESC
    """)
    rows = cur.fetchall()
    cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
    total_sync_ns = cur.fetchone()[0] or 1
    print(f"  Total sync time : {fmt_ns(total_sync_ns)}")
    print()
    print(f"  {'Sync Type':<35} {'Count':>10} {'Total':>12} {'Avg':>10} {'Max':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    for r in rows:
        print(f"  {str(r['sync_type']):<35} {r['count']:>10} "
              f"{fmt_ns(r['total_ns']):>12} {fmt_ns(r['avg_ns']):>10} {fmt_ns(r['max_ns']):>10}")

    # ------------------------------------------------------------------ #
    # 4. NVTX task breakdown (Legate tasks)
    # ------------------------------------------------------------------ #
    header("4. LEGATE TASK BREAKDOWN (NVTX Ranges)")
    cur.execute("""
        SELECT
            text,
            COUNT(*) AS calls,
            SUM(end - start) AS total_ns,
            AVG(end - start) AS avg_ns,
            MIN(end - start) AS min_ns,
            MAX(end - start) AS max_ns
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND text IS NOT NULL AND eventType = 59
        GROUP BY text
        ORDER BY total_ns DESC
        LIMIT ?
    """, (TOP_N,))
    rows = cur.fetchall()
    cur.execute("""
        SELECT SUM(end - start) FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND text IS NOT NULL AND eventType = 59
    """)
    total_nvtx_ns = cur.fetchone()[0] or 1
    print(f"  {'Task (NVTX)':<65} {'Calls':>6} {'Total':>10} {'%':>6} {'Avg':>10}")
    print(f"  {'-'*65} {'-'*6} {'-'*10} {'-'*6} {'-'*10}")
    for r in rows:
        name = str(r["text"])
        # Shorten path
        name = name.replace("/home/bod/profile/fuel_cell_3D/", "")
        name = name.replace("/home/bod/profile/fuel_cell_3D/./", "")
        name = name[:64]
        print(f"  {name:<65} {r['calls']:>6} {fmt_ns(r['total_ns']):>10} "
              f"{pct(r['total_ns'], total_nvtx_ns):>5.1f}% {fmt_ns(r['avg_ns']):>10}")

    # ------------------------------------------------------------------ #
    # 5. NVTX grouped by task type (ignore source location)
    # ------------------------------------------------------------------ #
    header("5. LEGATE TASK TYPE SUMMARY (grouped, no source location)")
    cur.execute("""
        SELECT
            CASE
                WHEN instr(text, ' : ') > 0 THEN substr(text, 1, instr(text, ' : ') - 1)
                ELSE text
            END AS task_type,
            COUNT(*) AS calls,
            SUM(end - start) AS total_ns,
            AVG(end - start) AS avg_ns
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND text IS NOT NULL AND eventType = 59
        GROUP BY task_type
        ORDER BY total_ns DESC
        LIMIT ?
    """, (TOP_N,))
    rows = cur.fetchall()
    print(f"  {'Task Type':<55} {'Calls':>7} {'Total':>12} {'%':>6} {'Avg':>10}")
    print(f"  {'-'*55} {'-'*7} {'-'*12} {'-'*6} {'-'*10}")
    for r in rows:
        name = str(r["task_type"])[:54]
        print(f"  {name:<55} {r['calls']:>7} {fmt_ns(r['total_ns']):>12} "
              f"{pct(r['total_ns'], total_nvtx_ns):>5.1f}% {fmt_ns(r['avg_ns']):>10}")

    # ------------------------------------------------------------------ #
    # 6. Time budget: kernel vs memcpy vs sync vs gap
    # ------------------------------------------------------------------ #
    header("6. TIME BUDGET SUMMARY")
    cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    kernel_total = cur.fetchone()[0] or 0
    cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    memcpy_total = cur.fetchone()[0] or 0
    cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
    sync_total = cur.fetchone()[0] or 0

    # OSRT (CPU-side blocking calls)
    cur.execute("""
        SELECT SUM(end - start) FROM OSRT_API
        WHERE end IS NOT NULL
    """)
    osrt_total = cur.fetchone()[0] or 0

    print(f"  GPU Kernels      : {fmt_ns(kernel_total):>12}  ({pct(kernel_total, total_wall):5.1f}% of wall time)")
    print(f"  GPU Memcpy       : {fmt_ns(memcpy_total):>12}  ({pct(memcpy_total, total_wall):5.1f}% of wall time)")
    print(f"  GPU Sync events  : {fmt_ns(sync_total):>12}  ({pct(sync_total, total_wall):5.1f}% of wall time)")
    print(f"  OS/CPU runtime   : {fmt_ns(osrt_total):>12}  ({pct(osrt_total, total_wall):5.1f}% of wall time)")
    print(f"  Wall time (total): {fmt_ns(total_wall):>12}")

    # ------------------------------------------------------------------ #
    # 7. Small kernel problem: histogram of kernel durations
    # ------------------------------------------------------------------ #
    header("7. KERNEL DURATION DISTRIBUTION (small kernel problem?)")
    thresholds = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, float("inf")]
    labels = ["<1us", "1-10us", "10-100us", "100us-1ms", "1ms-10ms", ">10ms"]
    buckets = defaultdict(lambda: [0, 0])  # [count, total_ns]
    cur.execute("SELECT end - start AS dur FROM CUPTI_ACTIVITY_KIND_KERNEL")
    for (dur,) in cur.fetchall():
        for i, thr in enumerate(thresholds[1:]):
            if dur < thr:
                buckets[labels[i]][0] += 1
                buckets[labels[i]][1] += dur
                break
    cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    total_kernels = cur.fetchone()[0]
    print(f"  Total kernels: {total_kernels}")
    print(f"  {'Duration bucket':<15} {'Count':>8} {'%Count':>8} {'Total time':>12} {'%KernelTime':>12}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*12} {'-'*12}")
    for lbl in labels:
        cnt, tns = buckets[lbl]
        print(f"  {lbl:<15} {cnt:>8} {pct(cnt, total_kernels):>7.1f}% "
              f"{fmt_ns(tns):>12} {pct(tns, kernel_total):>11.1f}%")

    # ------------------------------------------------------------------ #
    # 8. Top OSRT blocking calls (CPU side bottlenecks)
    # ------------------------------------------------------------------ #
    header("8. TOP CPU/OS BLOCKING CALLS")
    cur.execute("""
        SELECT
            s.value AS func_name,
            COUNT(*) AS calls,
            SUM(end - start) AS total_ns,
            AVG(end - start) AS avg_ns,
            MAX(end - start) AS max_ns
        FROM OSRT_API o
        JOIN StringIds s ON s.id = o.nameId
        WHERE end IS NOT NULL
        GROUP BY o.nameId
        ORDER BY total_ns DESC
        LIMIT ?
    """, (TOP_N,))
    rows = cur.fetchall()
    print(f"  {'Function':<40} {'Calls':>8} {'Total':>12} {'Avg':>10} {'Max':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
    for r in rows:
        print(f"  {str(r['func_name']):<40} {r['calls']:>8} "
              f"{fmt_ns(r['total_ns']):>12} {fmt_ns(r['avg_ns']):>10} {fmt_ns(r['max_ns']):>10}")

    # ------------------------------------------------------------------ #
    # 9. Memcpy bandwidth efficiency (large vs small transfers)
    # ------------------------------------------------------------------ #
    header("9. MEMCPY SIZE DISTRIBUTION (small transfer overhead?)")
    size_thresholds = [1024, 64*1024, 1024*1024, 16*1024*1024, float("inf")]
    size_labels = ["<1KB", "1KB-64KB", "64KB-1MB", "1MB-16MB", ">16MB"]
    sbuckets = defaultdict(lambda: [0, 0, 0])  # count, bytes, ns
    cur.execute("SELECT bytes, end - start FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    for (b, dur) in cur.fetchall():
        for i, thr in enumerate(size_thresholds):
            if b < thr:
                sbuckets[size_labels[i]][0] += 1
                sbuckets[size_labels[i]][1] += b
                sbuckets[size_labels[i]][2] += dur
                break
    cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    total_mc = cur.fetchone()[0]
    print(f"  {'Size bucket':<15} {'Count':>7} {'%Count':>7} {'Bytes':>12} {'Time':>10} {'BW GB/s':>9}")
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*12} {'-'*10} {'-'*9}")
    for lbl in size_labels:
        cnt, tbytes, tns = sbuckets[lbl]
        bw = tbytes / tns if tns > 0 else 0
        print(f"  {lbl:<15} {cnt:>7} {pct(cnt, total_mc):>6.1f}% "
              f"{fmt_bytes(tbytes):>12} {fmt_ns(tns):>10} {bw:>9.1f}")

    # ------------------------------------------------------------------ #
    # 10. GPU utilization timeline (coarse 1-second buckets)
    # ------------------------------------------------------------------ #
    header("10. GPU UTILIZATION TIMELINE (1s buckets)")
    bucket_ns = 1_000_000_000  # 1 second
    cur.execute("""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY start
    """)
    kernels = cur.fetchall()
    if kernels:
        util_buckets = defaultdict(int)
        for s, e in kernels:
            b_start = (s - t0) // bucket_ns
            b_end   = (e - t0) // bucket_ns
            if b_start == b_end:
                util_buckets[b_start] += e - s
            else:
                util_buckets[b_start] += (b_start + 1) * bucket_ns + t0 - s
                for b in range(b_start + 1, b_end):
                    util_buckets[b] += bucket_ns
                util_buckets[b_end] += e - b_end * bucket_ns - t0
        n_buckets = (t1 - t0) // bucket_ns + 1
        print(f"  Bucket  Utilization  Bar")
        print(f"  {'---':<6}  {'---':>11}  {'---'}")
        for b in range(int(n_buckets)):
            u = pct(util_buckets[b], bucket_ns)
            bar = "#" * int(u / 2)
            print(f"  {b:>4}s   {u:>9.1f}%  {bar}")

    # ------------------------------------------------------------------ #
    # 11. Profiler overhead
    # ------------------------------------------------------------------ #
    header("11. PROFILER / CUPTI OVERHEAD")
    cur.execute("""
        SELECT
            e.label AS overhead_type,
            COUNT(*) AS count,
            SUM(end - start) AS total_ns
        FROM CUPTI_ACTIVITY_KIND_OVERHEAD o
        LEFT JOIN ENUM_CUPTI_OVERHEAD_TYPE e ON e.id = o.overheadType
        GROUP BY o.overheadType
        ORDER BY total_ns DESC
    """)
    rows = cur.fetchall()
    for r in rows:
        print(f"  {str(r['overhead_type']):<30} count={r['count']:>8}  time={fmt_ns(r['total_ns'])}")

    # ------------------------------------------------------------------ #
    # 12. Bottleneck summary
    # ------------------------------------------------------------------ #
    header("12. BOTTLENECK SUMMARY & RECOMMENDATIONS")

    cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end - start < 10000")
    tiny_kernels = cur.fetchone()[0]
    tiny_pct = pct(tiny_kernels, total_kernels)

    cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE bytes < 65536")
    small_mc = cur.fetchone()[0]
    small_mc_pct = pct(small_mc, total_mc)

    cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
    sync_count = cur.fetchone()[0]

    issues = []
    if tiny_pct > 20:
        issues.append(f"[!] SMALL KERNELS: {tiny_pct:.1f}% of kernels run <10us — "
                      f"high launch overhead. Consider fusing ops or batching tasks.")
    if small_mc_pct > 30:
        issues.append(f"[!] SMALL MEMCPY: {small_mc_pct:.1f}% of transfers are <64KB — "
                      f"consider batching or using pinned memory / async transfers.")
    if sync_count > 1_000_000:
        issues.append(f"[!] EXCESSIVE SYNC: {sync_count:,} synchronization events — "
                      f"too many stream syncs or cudaDeviceSynchronize calls block the GPU pipeline.")
    if pct(memcpy_total, total_wall) > 20:
        issues.append(f"[!] MEMCPY BOTTLENECK: memory copies take "
                      f"{pct(memcpy_total, total_wall):.1f}% of wall time — "
                      f"check Host↔Device transfers, prefer device-resident data.")
    if pct(kernel_total, total_wall) < 50:
        issues.append(f"[!] LOW GPU UTILIZATION: GPU kernels only cover "
                      f"{pct(kernel_total, total_wall):.1f}% of wall time — "
                      f"CPU/scheduling overhead is the likely bottleneck.")

    if issues:
        for iss in issues:
            print(f"  {iss}")
    else:
        print("  No obvious single bottleneck detected. Profile looks balanced.")

    print()
    conn.close()
    print(f"  Analysis complete. DB: {DB_PATH}")

if __name__ == "__main__":
    run()
