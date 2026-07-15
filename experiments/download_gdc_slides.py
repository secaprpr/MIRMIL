"""Query and download public GDC diagnostic slide files.

The script is intentionally small and dependency-light so it can run on servers
without the official ``gdc-client`` binary. It supports two stages:

1. query a reproducible TSV manifest from the GDC files API;
2. download each file by UUID with size-based resume/skip behavior.

It is designed for public TCGA SVS slides. It does not delete partial files.
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data/{file_id}"


MANIFEST_COLUMNS = [
    "id",
    "filename",
    "md5",
    "size",
    "state",
    "project_id",
    "case_submitter_id",
    "data_format",
    "experimental_strategy",
    "sample_type",
    "slide_submitter_id",
]


def build_filters(projects):
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": projects,
                },
            },
            {
                "op": "in",
                "content": {"field": "files.data_format", "value": ["SVS"]},
            },
            {
                "op": "in",
                "content": {
                    "field": "files.experimental_strategy",
                    "value": ["Diagnostic Slide"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.cases.samples.sample_type",
                    "value": ["Primary Tumor"],
                },
            },
        ],
    }


def first_nested(values, default=""):
    if not values:
        return default
    return values[0]


def normalize_hit(hit):
    cases = hit.get("cases") or []
    case = first_nested(cases, {})
    samples = case.get("samples") or []
    sample = first_nested(samples, {})
    portions = sample.get("portions") or []
    portion = first_nested(portions, {})
    slides = portion.get("slides") or []
    slide = first_nested(slides, {})
    return {
        "id": hit.get("file_id") or hit.get("id") or "",
        "filename": hit.get("file_name") or "",
        "md5": hit.get("md5sum") or "",
        "size": int(hit.get("file_size") or 0),
        "state": hit.get("state") or "",
        "project_id": (case.get("project") or {}).get("project_id") or "",
        "case_submitter_id": case.get("submitter_id") or "",
        "data_format": hit.get("data_format") or "",
        "experimental_strategy": hit.get("experimental_strategy") or "",
        "sample_type": sample.get("sample_type") or "",
        "slide_submitter_id": slide.get("submitter_id") or "",
    }


def query_manifest(projects, output, page_size=2000, timeout=120):
    fields = ",".join(
        [
            "file_id",
            "file_name",
            "md5sum",
            "file_size",
            "state",
            "cases.project.project_id",
            "cases.submitter_id",
            "cases.samples.sample_type",
            "cases.samples.portions.slides.submitter_id",
            "experimental_strategy",
            "data_format",
        ]
    )
    params = {
        "filters": json.dumps(build_filters(projects)),
        "fields": fields,
        "format": "JSON",
        "size": str(page_size),
        "from": "0",
        "sort": "file_name:asc",
    }
    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()["data"]
    total = int(payload["pagination"]["total"])
    rows = []
    while len(rows) < total:
        if rows:
            params["from"] = str(len(rows))
            response = requests.get(
                GDC_FILES_ENDPOINT, params=params, timeout=timeout
            )
            response.raise_for_status()
            payload = response.json()["data"]
        rows.extend(normalize_hit(hit) for hit in payload["hits"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    total_size = sum(row["size"] for row in rows)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "projects": projects,
                "files": len(rows),
                "bytes": total_size,
                "gib": round(total_size / 1024**3, 2),
            },
            indent=2,
        )
    )


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def download_one(session, row, output_dir, chunk_size, timeout, retries):
    file_id = row["id"]
    filename = row["filename"]
    expected_size = int(row["size"])
    target_dir = output_dir / file_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.exists() and target.stat().st_size == expected_size:
        return "skip_complete", target
    if target.exists() and target.stat().st_size > expected_size:
        raise RuntimeError(
            f"Local file is larger than manifest size: {target} "
            f"{target.stat().st_size}>{expected_size}"
        )

    headers = {}
    mode = "wb"
    existing_size = target.stat().st_size if target.exists() else 0
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    url = GDC_DATA_ENDPOINT.format(file_id=file_id)
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 416:
                    if target.exists() and target.stat().st_size == expected_size:
                        return "skip_complete", target
                r.raise_for_status()
                if existing_size and r.status_code != 206:
                    # Server did not honor the range request; restart safely.
                    mode = "wb"
                with target.open(mode + "") as handle:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
            if target.stat().st_size == expected_size:
                return "downloaded", target
            headers = {}
            existing_size = target.stat().st_size
            if existing_size < expected_size:
                headers["Range"] = f"bytes={existing_size}-"
                mode = "ab"
            raise RuntimeError(
                f"size mismatch after download: {target.stat().st_size} "
                f"!= {expected_size}"
            )
        except Exception as exc:
            if attempt == retries:
                raise
            print(
                f"[retry {attempt}/{retries}] {filename}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"unreachable download failure for {filename}")


def download_manifest(
    manifest, output_dir, chunk_size, timeout, retries, limit, workers
):
    rows = read_manifest(manifest)
    if limit:
        rows = rows[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"skip_complete": 0, "downloaded": 0}
    indexed_rows = list(enumerate(rows, start=1))

    def run(index_and_row):
        index, row = index_and_row
        session = requests.Session()
        status, target = download_one(
            session=session,
            row=row,
            output_dir=output_dir,
            chunk_size=chunk_size,
            timeout=timeout,
            retries=retries,
        )
        return index, row, status, target

    if workers <= 1:
        iterator = map(run, indexed_rows)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        iterator = as_completed(
            [executor.submit(run, item) for item in indexed_rows]
        )

    try:
        for item in iterator:
            if workers > 1:
                index, row, status, target = item.result()
            else:
                index, row, status, target = item
            counts[status] = counts.get(status, 0) + 1
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(rows),
                        "status": status,
                        "file": str(target),
                        "size": int(row["size"]),
                    }
                ),
                flush=True,
            )
    finally:
        if workers > 1:
            executor.shutdown(wait=False, cancel_futures=True)
    print(json.dumps({"done": True, "counts": counts}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser("query")
    query.add_argument("--project", action="append", required=True)
    query.add_argument("--output", type=Path, required=True)
    query.add_argument("--page-size", type=int, default=2000)
    query.add_argument("--timeout", type=int, default=120)

    download = subparsers.add_parser("download")
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    download.add_argument("--timeout", type=int, default=300)
    download.add_argument("--retries", type=int, default=5)
    download.add_argument("--limit", type=int, default=0)
    download.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()
    if args.command == "query":
        query_manifest(
            projects=args.project,
            output=args.output,
            page_size=args.page_size,
            timeout=args.timeout,
        )
    elif args.command == "download":
        download_manifest(
            manifest=args.manifest,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            timeout=args.timeout,
            retries=args.retries,
            limit=args.limit,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
