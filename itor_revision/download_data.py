"""Download selected release archives and verify their SHA-256 checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phases", nargs="*", help="phase names, or all")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "downloads")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("DATASETS.json").read_text())
    assets = {item["phase"]: item for item in manifest["assets"]}
    if args.list or not args.phases:
        for phase, item in assets.items():
            print(f"{phase:25s} {item['bytes'] / 1048576:9.2f} MiB")
        return
    names = list(assets) if args.phases == ["all"] else args.phases
    unknown = set(names) - set(assets)
    if unknown:
        parser.error("Unknown phases: " + ", ".join(sorted(unknown)))
    args.out.mkdir(parents=True, exist_ok=True)
    for phase in names:
        item = assets[phase]
        destination = args.out / item["name"]
        if destination.exists():
            if destination.stat().st_size == item["bytes"] and digest(destination) == item["sha256"]:
                print("Already verified:", destination)
                continue
            raise RuntimeError("Existing archive has a different checksum: " + str(destination))
        partial = destination.with_suffix(".zip.part")
        request = urllib.request.Request(item["url"], headers={"User-Agent": "layout-reproducibility"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if partial.stat().st_size != item["bytes"] or digest(partial) != item["sha256"]:
            raise RuntimeError("Checksum mismatch: " + str(partial))
        partial.replace(destination)
        print("Downloaded and verified:", destination, flush=True)
    print("Extract each ZIP into the same data root; archive paths include the phase folder.")


if __name__ == "__main__":
    main()
