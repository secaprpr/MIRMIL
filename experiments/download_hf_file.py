import argparse
import os
import time
from pathlib import Path

import requests
from huggingface_hub import get_token, hf_hub_url


def remote_size(response):
    linked_size = response.headers.get("X-Linked-Size")
    if linked_size is not None:
        return int(linked_size)
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    content_length = response.headers.get("Content-Length")
    return int(content_length) if content_length is not None else None


def download_file(
    repo_id,
    filename,
    destination,
    repo_type="dataset",
    retries=20,
    chunk_size=8 * 1024 * 1024,
):
    token = get_token()
    if not token:
        raise RuntimeError("No Hugging Face token is configured")
    headers = {"Authorization": f"Bearer {token}"}
    url = hf_hub_url(repo_id, filename, repo_type=repo_type)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request_headers = dict(headers)
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(
                url,
                headers=request_headers,
                stream=True,
                timeout=(30, 300),
                allow_redirects=True,
            ) as response:
                if response.status_code not in (200, 206):
                    response.raise_for_status()
                expected_size = remote_size(response)
                if offset and response.status_code == 200:
                    offset = 0
                    partial.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                with open(partial, mode) as output:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            output.write(chunk)
                actual_size = partial.stat().st_size
                if expected_size is not None and actual_size != expected_size:
                    raise IOError(
                        f"Expected {expected_size} bytes, got {actual_size}"
                    )
                os.replace(partial, destination)
                return destination
        except (OSError, requests.RequestException) as error:
            if attempt == retries:
                raise
            print(
                f"attempt {attempt}/{retries} failed at byte {offset}: "
                f"{error}"
            )
            time.sleep(min(5 * attempt, 60))
    raise RuntimeError("Download retries exhausted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument(
        "--resume-from",
        help="Existing partial file to move into the managed resume path",
    )
    args = parser.parse_args()

    destination = Path(args.destination)
    partial = destination.with_name(destination.name + ".partial")
    if args.resume_from:
        resume_from = Path(args.resume_from)
        if partial.exists() and partial.resolve() != resume_from.resolve():
            raise FileExistsError(partial)
        if not partial.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(resume_from, partial)
    result = download_file(
        args.repo_id,
        args.filename,
        destination,
        repo_type=args.repo_type,
        retries=args.retries,
    )
    print(f"downloaded={result} bytes={result.stat().st_size}")


if __name__ == "__main__":
    main()
