#!/usr/bin/env python3
"""Minimal S3 public-bucket lister (no AWS CLI needed)."""
import re
import sys
import urllib.request

BUCKET = "https://cellpainting-gallery.s3.amazonaws.com"


def list_prefix(prefix, max_keys=1000):
    url = f"{BUCKET}?list-type=2&prefix={prefix}&max-keys={max_keys}"
    with urllib.request.urlopen(url, timeout=120) as r:
        xml = r.read().decode("utf-8")
    rows = []
    for m in re.finditer(r"<Contents>(.*?)</Contents>", xml, re.S):
        b = m.group(1)
        key = re.search(r"<Key>(.*?)</Key>", b).group(1)
        size = int(re.search(r"<Size>(\d+)</Size>", b).group(1))
        rows.append((key, size))
    return rows


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = list_prefix(prefix)
    total = 0
    for key, size in rows:
        if size == 0:
            continue
        total += size
        print(f"{size/1e6:10.2f} MB  {key}")
    print(f"\nTOTAL: {total/1e6:.1f} MB  ({len([r for r in rows if r[1]>0])} files)")


if __name__ == "__main__":
    main()
