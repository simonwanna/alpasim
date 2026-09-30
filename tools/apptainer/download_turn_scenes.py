# SPDX-License-Identifier: Apache-2.0
"""Download exactly three labelled scene candidates with a 7.5 GiB archive cap."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

os.umask(0o007)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--catalog", type=Path, required=True)
parser.add_argument("--assets", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--timezone", default="UTC")
args = parser.parse_args()
repo = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
rev = "ebbb8d5b433bcf072451a55e3d8b86c5ed7a9396"
assets = args.assets
stamp = datetime.now(ZoneInfo(args.timezone)).strftime("%Y-%m-%d/%H-%M-%S%z")
out = args.output_root / stamp
out.mkdir(parents=True, exist_ok=False, mode=0o2770)
log = (out / "scene-download.log").open("x")


def say(*args):
    line = " ".join(map(str, args))
    print(line, flush=True)
    print(line, file=log, flush=True)


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise RuntimeError("Refusing non-HTTPS redirect")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlsplit(newurl).netloc != "huggingface.co":
            new.remove_header("Authorization")
        return new


token_path = Path(
    os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token"))
)
token = os.environ.get("HF_TOKEN", "").strip() or token_path.read_text().strip()
assert token, "No Hugging Face login found"
opener = build_opener(SafeRedirect())


def fetch(url):
    assert urlsplit(url).netloc == "huggingface.co"
    return opener.open(
        Request(url, headers={"Authorization": "Bearer " + token}), timeout=120
    )


def read_json(url, limit):
    with fetch(url) as response:
        data = response.read(limit + 1)
    assert len(data) <= limit, "Metadata exceeds size limit"
    return json.loads(data)


def strings(value):
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, (dict, list)):
        return " ".join(
            strings(v) for v in (value.values() if isinstance(value, dict) else value)
        )
    return ""


say("Log:", out / "scene-download.log")
info = read_json(
    f"https://huggingface.co/api/datasets/{repo}/revision/{rev}?blobs=true", 32 * 2**20
)
assert info["sha"] == rev
files = {f["rfilename"]: f for f in info["siblings"]}
catalog = args.catalog
with (catalog / "sim_suites.csv").open() as f:
    ids = {
        r["uuid"]
        for r in csv.DictReader(f)
        if r["test_suite_id"] == "public_2601_video_model"
    }
with (catalog / "sim_scenes.csv").open() as f:
    rows = sorted(
        (r for r in csv.DictReader(f) if r["uuid"] in ids), key=lambda r: r["path"]
    )
selected, checked = [], 0
for row in rows:
    name = row["path"]
    label = str(Path(name).parent / "labels.json")
    size = files.get(name, {}).get("size", 0)
    if label not in files or not 0 < size <= 2.5 * 2**30:
        continue
    labels = read_json(
        f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{label}", 256 * 2**10
    )
    checked += 1
    text = strings(labels)
    if any(word in text for word in ("intersection", "junction", "turn")):
        if row["scene_id"] not in {s["scene_id"] for s in selected}:
            digest = files[name].get("lfs", {}).get("sha256")
            assert digest and len(digest) == 64, "Missing SHA256 for scene"
            selected.append(dict(row, bytes=size, sha256=digest, labels=labels))
            say(
                "SELECT:",
                row["scene_id"],
                f"{size / 2**30:.2f} GiB",
                json.dumps(labels)[:500],
            )
    if len(selected) == 3 or checked == 128:
        break
    if checked % 16 == 0:
        say("Labels checked:", checked, "candidates:", len(selected))
assert len(selected) == 3, (
    "Fewer than 3 labelled candidates; no scene archives downloaded"
)
total = sum(s["bytes"] for s in selected)
assert total <= 7.5 * 2**30
assert not assets.is_symlink()
assets.mkdir(parents=True, exist_ok=True, mode=0o2770)
assert shutil.disk_usage(assets).free > total + 2**30, (
    "Insufficient filesystem free space"
)
manifest = dict(
    repo=repo,
    revision=rev,
    selection="Label-based candidates; junction suitability unverified",
    scenes=selected,
)
(out / "scene-selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
say(
    "Downloading:",
    len(selected),
    "scenes;",
    f"{total / 2**30:.2f} GiB total;",
    "to",
    assets,
)

for scene in selected:
    dest = assets / Path(scene["path"]).name
    assert not dest.is_symlink()
    if dest.exists():
        with dest.open("rb") as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(8 * 2**20), b""):
                digest.update(block)
        assert (
            dest.stat().st_size == scene["bytes"]
            and digest.hexdigest() == scene["sha256"]
        ), f"Existing file differs: {dest}"
        say("REUSE:", dest.name)
        continue
    part = dest.with_suffix(".usdz.part")
    assert not part.is_symlink()
    assert not part.exists() or part.stat().st_uid == os.getuid(), part
    digest, count, reported = hashlib.sha256(), 0, 0
    with (
        fetch(
            f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{scene['path']}"
        ) as response,
        part.open("wb") as f,
    ):
        while True:
            block = response.read(8 * 2**20)
            if not block:
                break
            count += len(block)
            assert count <= scene["bytes"], "Download exceeds expected size"
            f.write(block)
            digest.update(block)
            if count - reported >= 512 * 2**20:
                say(dest.name, f"{count / 2**30:.2f}/{scene['bytes'] / 2**30:.2f} GiB")
                reported = count
    assert count == scene["bytes"] and digest.hexdigest() == scene["sha256"], (
        "Size/hash verification failed"
    )
    # Atomic no-overwrite publication; the temporary file is on the same filesystem.
    os.link(part, dest)
    part.unlink()
    say("PASS:", dest.name, "size and SHA256 verified")
say("DONE. Selection manifest:", out / "scene-selection.json")
