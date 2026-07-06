#!/usr/bin/env python3
"""초대용량 CSV(수십 GB)에서 cluster_id 별 대표 행을 병렬로 추출하는 스크립트.

extract_cluster_samples.py 와 같은 일을 하지만, 파일을 바이트 구간으로
나눠 여러 프로세스가 동시에 스캔한다. 각 워커는 자기 구간에서 클러스터별
후보 행과 바이트 오프셋만 모아 반환하므로(클러스터 수만큼, 수 KB 수준)
메모리 사용량은 파일 크기와 무관하게 거의 일정하다.

속도를 위해 행 전체를 파싱하지 않고 cluster_id 필드만 잘라내며,
이미 본 클러스터의 행은 즉시 건너뛴다. 결과는 순차 버전과
바이트 단위로 동일하다 (first 모드 기준).

제약: 따옴표로 감싼 필드 안에 줄바꿈(\n)이 들어간 CSV는 지원하지 않는다.
(숫자/해시 위주 데이터에서는 해당 없음. 따옴표 안의 쉼표는 지원한다.)

사용법:
    python3 extract_cluster_samples_fast.py input.csv -o output.csv
    python3 extract_cluster_samples_fast.py input.csv -o output.csv -j 16
    python3 extract_cluster_samples_fast.py input.csv --mode random --seed 42
"""

import argparse
import csv
import os
import random
import sys
import time
from multiprocessing import Pool

BLOCK_SIZE = 1 << 24  # 워커당 한 번에 읽는 크기 (16 MiB)
MIN_CHUNK = 4 << 20   # 이보다 작은 구간은 워커를 늘리지 않음 (4 MiB)


def _extract_key(line, key_idx):
    """CSV 한 행(bytes, 개행 제거됨)에서 key_idx 번째 필드를 뽑는다.

    따옴표가 없는 행은 단순 split(빠른 경로), 따옴표가 있으면 csv 모듈로
    정확히 파싱한다. 필드 수가 모자란 행은 None을 반환해 건너뛴다.
    """
    if b'"' not in line:
        parts = line.split(b",", key_idx + 1)
        if len(parts) <= key_idx:
            return None
        return parts[key_idx]
    try:
        row = next(csv.reader([line.decode("utf-8", "replace")]))
    except (csv.Error, StopIteration):
        return None
    if len(row) <= key_idx:
        return None
    return row[key_idx].encode("utf-8")


def scan_chunk(task):
    """파일의 [start, end) 바이트 구간을 스캔해 클러스터별 후보를 모은다.

    구간 경계에 걸친 행은 '행이 시작된 구간'이 처리한다. 반환값은
    {key: (first_off, last_off, line, count)} 형태로, 병합 단계에서
    모드별 선택과 등장 순서 복원에 쓰인다.
    """
    path, start, end, data_start, key_idx, mode, seed, chunk_no = task
    result = {}
    rng = random.Random(seed * 1_000_003 + chunk_no) if seed is not None else random.Random()
    rows = 0

    with open(path, "rb") as f:
        if start == data_start:
            f.seek(start)
        else:
            # start 직전 바이트가 \n이면 start는 행의 시작이므로 그대로 진행,
            # 아니면 이전 구간에서 시작된 행의 잔여분을 버린다.
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()
        pos = f.tell()

        done = False
        while not done:
            block = f.read(BLOCK_SIZE)
            if not block:
                break
            if not block.endswith(b"\n"):
                block += f.readline()  # 블록 끝의 잘린 행을 마저 읽는다
            lines = block.split(b"\n")
            if lines and lines[-1] == b"":
                lines.pop()
            for line in lines:
                if pos >= end:
                    done = True
                    break
                line_len = len(line) + 1
                if line.endswith(b"\r"):
                    line = line[:-1]
                if line:
                    rows += 1
                    key = _extract_key(line, key_idx)
                    if key is not None:
                        entry = result.get(key)
                        if entry is None:
                            result[key] = (pos, pos, line, 1)
                        elif mode == "last":
                            result[key] = (entry[0], pos, line, entry[3] + 1)
                        elif mode == "random":
                            count = entry[3] + 1
                            if rng.random() < 1.0 / count:
                                result[key] = (entry[0], pos, line, count)
                            else:
                                result[key] = (entry[0], entry[1], entry[2], count)
                        # mode == "first": 첫 후보 유지, 아무것도 안 함
                pos += line_len

    return chunk_no, rows, result


def merge_results(chunk_results, mode, seed):
    """워커별 부분 결과를 합쳐 클러스터당 최종 한 행을 고른다."""
    merged = {}  # key -> [order_off, pick_off, line, count]
    rng = random.Random(seed * 1_000_003 - 1) if seed is not None else random.Random()

    for _, _, result in sorted(chunk_results):  # chunk_no 순서로 병합
        for key, (first_off, off, line, count) in result.items():
            entry = merged.get(key)
            if entry is None:
                merged[key] = [first_off, off, line, count]
                continue
            entry[0] = min(entry[0], first_off)
            if mode == "first":
                if first_off < entry[1]:
                    entry[1], entry[2] = first_off, line
            elif mode == "last":
                if off > entry[1]:
                    entry[1], entry[2] = off, line
            elif mode == "random":
                total = entry[3] + count
                if rng.random() < count / total:
                    entry[1], entry[2] = off, line
                entry[3] = total

    # 클러스터가 파일에서 처음 등장한 순서대로 출력
    return [line for _, _, line, _ in sorted(merged.values())], merged


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="대용량 CSV에서 cluster_id 별 행을 하나씩 병렬로 추출한다."
    )
    parser.add_argument("input", help="입력 CSV 경로")
    parser.add_argument(
        "-o", "--output", default="cluster_samples.csv",
        help="출력 CSV 경로 (기본값: cluster_samples.csv)",
    )
    parser.add_argument(
        "-k", "--key-column", default="cluster_id",
        help="그룹 기준 컬럼 이름 (기본값: cluster_id)",
    )
    parser.add_argument(
        "-m", "--mode", choices=["first", "last", "random"], default="random",
        help="클러스터 내 행 선택 방식 (기본값: random — 입력이 좌표순으로 "
             "정렬돼 있어도 편향 없이 뽑힌다)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="--mode random 일 때 재현용 시드 (기본값: 42)",
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=None,
        help="병렬 프로세스 수 (기본값: CPU 코어 수)",
    )
    args = parser.parse_args(argv)

    with open(args.input, "rb") as f:
        header_line = f.readline()
        data_start = f.tell()
    if not header_line.strip():
        sys.exit(f"오류: 입력 파일이 비어 있습니다: {args.input}")
    fieldnames = next(csv.reader([header_line.decode("utf-8-sig")]))
    if args.key_column not in fieldnames:
        sys.exit(
            f"오류: '{args.key_column}' 컬럼이 없습니다. "
            f"존재하는 컬럼: {', '.join(fieldnames)}"
        )
    key_idx = fieldnames.index(args.key_column)

    size = os.path.getsize(args.input)
    span = size - data_start
    workers = args.workers or os.cpu_count() or 1
    workers = max(1, min(workers, span // MIN_CHUNK or 1))

    starts = [data_start + span * i // workers for i in range(workers)] + [size]
    tasks = [
        (args.input, starts[i], starts[i + 1], data_start,
         key_idx, args.mode, args.seed, i)
        for i in range(workers)
    ]

    t0 = time.monotonic()
    chunk_results = []
    if workers == 1:
        chunk_results.append(scan_chunk(tasks[0]))
    else:
        with Pool(workers) as pool:
            for done, res in enumerate(pool.imap_unordered(scan_chunk, tasks), 1):
                chunk_results.append(res)
                print(f"\r구간 {done}/{workers} 완료", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    elapsed = time.monotonic() - t0

    selected, merged = merge_results(chunk_results, args.mode, args.seed)
    total_rows = sum(rows for _, rows, _ in chunk_results)

    with open(args.output, "wb") as f:
        f.write(header_line.rstrip(b"\r\n") + b"\n")
        for line in selected:
            f.write(line + b"\n")

    mb = span / (1 << 20)
    rate = mb / elapsed if elapsed > 0 else 0.0
    print(
        f"전체 {total_rows:,}행({mb:,.0f} MiB)에서 클러스터 {len(selected)}개의 "
        f"대표 행을 추출했습니다."
    )
    print(f"워커 {workers}개, {elapsed:.1f}초 소요 ({rate:,.0f} MiB/s)")
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
