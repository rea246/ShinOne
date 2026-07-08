#!/usr/bin/env python3
"""초대용량 CSV(수십 GB)에서 cluster_id 별 대표 행을 병렬로 추출한다.

파일을 바이트 구간으로 나눠 여러 프로세스가 동시에 스캔한다.
행 전체를 파싱하지 않고 키 필드만 잘라내므로 빠르고, 메모리 사용량은
파일 크기와 무관하게 거의 일정하다(클러스터별 후보 행만 유지).

설정은 main() 상단의 변수로 지정한다.

제약: 따옴표 필드 안에 줄바꿈이 있는 CSV는 지원하지 않는다
(따옴표 안의 쉼표는 지원).
"""

import csv
import os
import random
import sys
import time
from multiprocessing import Pool

BLOCK_SIZE = 1 << 24  # 워커당 한 번에 읽는 크기 (16 MiB)
MIN_CHUNK = 4 << 20   # 이보다 작은 구간은 워커를 늘리지 않음 (4 MiB)


def _extract_key(line, key_idx):
    """행(bytes)에서 key_idx 번째 필드를 뽑는다. 따옴표 행은 csv 모듈로 파싱."""
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
    """[start, end) 바이트 구간을 스캔해 {key: (first_off, last_off, line, count)} 반환.

    구간 경계에 걸친 행은 행이 시작된 구간이 처리한다.
    """
    path, start, end, data_start, key_idx, mode, seed, chunk_no = task
    result = {}
    rng = random.Random(seed * 1_000_003 + chunk_no) if seed is not None else random.Random()
    rows = 0

    with open(path, "rb") as f:
        if start == data_start:
            f.seek(start)
        else:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()  # 이전 구간에서 시작된 행의 잔여분을 버린다
        pos = f.tell()

        done = False
        while not done:
            block = f.read(BLOCK_SIZE)
            if not block:
                break
            if not block.endswith(b"\n"):
                block += f.readline()
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
                pos += line_len

    return chunk_no, rows, result


def merge_results(chunk_results, mode, seed):
    """워커별 부분 결과를 합쳐 클러스터당 최종 한 행을 고른다."""
    merged = {}  # key -> [order_off, pick_off, line, count]
    rng = random.Random(seed * 1_000_003 - 1) if seed is not None else random.Random()

    for _, _, result in sorted(chunk_results):
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
    return [line for _, _, line, _ in sorted(merged.values())]


def extract(input_path, output_path, key_column, mode, seed, workers):
    with open(input_path, "rb") as f:
        header_line = f.readline()
        data_start = f.tell()
    if not header_line.strip():
        sys.exit(f"오류: 입력 파일이 비어 있습니다: {input_path}")
    fieldnames = next(csv.reader([header_line.decode("utf-8-sig")]))
    if key_column not in fieldnames:
        sys.exit(f"오류: '{key_column}' 컬럼이 없습니다. 존재하는 컬럼: {', '.join(fieldnames)}")
    key_idx = fieldnames.index(key_column)

    size = os.path.getsize(input_path)
    span = size - data_start
    workers = workers or os.cpu_count() or 1
    workers = max(1, min(workers, span // MIN_CHUNK or 1))

    starts = [data_start + span * i // workers for i in range(workers)] + [size]
    tasks = [
        (input_path, starts[i], starts[i + 1], data_start, key_idx, mode, seed, i)
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

    selected = merge_results(chunk_results, mode, seed)
    total_rows = sum(rows for _, rows, _ in chunk_results)

    with open(output_path, "wb") as f:
        f.write(header_line.rstrip(b"\r\n") + b"\n")
        for line in selected:
            f.write(line + b"\n")

    mb = span / (1 << 20)
    rate = mb / elapsed if elapsed > 0 else 0.0
    print(f"전체 {total_rows:,}행({mb:,.0f} MiB)에서 클러스터 {len(selected)}개의 대표 행을 추출했습니다.")
    print(f"워커 {workers}개, {elapsed:.1f}초 소요 ({rate:,.0f} MiB/s)")
    print(f"저장 완료: {output_path}")


def main():
    input_path = "input.csv"          # 입력 CSV
    output_path = "B.csv"             # 출력 CSV
    key_column = "cluster_id"         # 그룹 기준 컬럼
    mode = "random"                   # first / last / random
    seed = 42                         # random 모드 재현용 시드 (None 이면 매번 다름)
    workers = None                    # 병렬 프로세스 수 (None 이면 CPU 코어 수)

    extract(input_path, output_path, key_column, mode, seed, workers)


if __name__ == "__main__":
    main()
