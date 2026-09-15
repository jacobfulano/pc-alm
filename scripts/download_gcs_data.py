from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a versioned GCS dataset prefix."
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    blobs = [
        blob
        for blob in client.list_blobs(args.bucket, prefix=args.prefix)
        if blob.name.endswith(".gz")
    ]
    if not blobs:
        raise FileNotFoundError(
            f"no .gz objects found at gs://{args.bucket}/{args.prefix}"
        )
    for blob in blobs:
        destination = args.output_dir / Path(blob.name).name
        blob.download_to_filename(destination)
        print(f"downloaded gs://{args.bucket}/{blob.name} -> {destination}", flush=True)


if __name__ == "__main__":
    main()
