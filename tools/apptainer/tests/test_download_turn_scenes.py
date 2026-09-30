import csv
import hashlib
import io
import json
import runpy
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "download_turn_scenes.py"
REVISION = "ebbb8d5b433bcf072451a55e3d8b86c5ed7a9396"


@pytest.fixture
def download(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    rows = [
        dict(uuid=str(i), scene_id=f"clipgt-{i}", path=f"scenes/{i}/{i}.usdz")
        for i in range(4)
    ]
    for filename, fields, entries in [
        ("sim_scenes.csv", ["uuid", "scene_id", "path"], rows),
        (
            "sim_suites.csv",
            ["uuid", "test_suite_id"],
            [
                dict(uuid=r["uuid"], test_suite_id="public_2601_video_model")
                for r in rows
            ],
        ),
    ]:
        with (catalog / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)
    state = dict(archives=[], eligible=4, corrupt=False, oversized=False)
    data = b"fake scene data"
    siblings = []
    for row in rows:
        siblings.extend(
            [
                dict(
                    rfilename=row["path"],
                    size=len(data),
                    lfs=dict(sha256=hashlib.sha256(data).hexdigest()),
                ),
                dict(rfilename=str(Path(row["path"]).parent / "labels.json"), size=40),
            ]
        )

    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Authorization") == "Bearer fake-test-token"
            url = request.full_url
            if "/api/" in url:
                if state["oversized"]:
                    for entry in siblings:
                        if entry["rfilename"].endswith(".usdz"):
                            entry["size"] = 3 * 2**30
                return io.BytesIO(
                    json.dumps(dict(sha=REVISION, siblings=siblings)).encode()
                )
            if url.endswith("labels.json"):
                number = int(url.split("/")[-2])
                return io.BytesIO(
                    json.dumps(
                        dict(
                            behavior="right turn"
                            if number < state["eligible"]
                            else "lane keeping"
                        )
                    ).encode()
                )
            state["archives"].append(url)
            return io.BytesIO(b"X" * len(data) if state["corrupt"] else data)

    monkeypatch.setattr("urllib.request.build_opener", lambda *a: Opener())
    monkeypatch.setenv("HF_TOKEN", "fake-test-token")

    def invoke():
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--catalog",
                str(catalog),
                "--assets",
                str(tmp_path / "assets"),
                "--output-root",
                str(tmp_path / "outputs"),
            ],
        )
        return runpy.run_path(str(SCRIPT), run_name="__main__")

    return invoke, state, tmp_path


def test_downloads_exactly_three_and_records_selection(download):
    invoke, state, root = download
    namespace = invoke()
    assert len(state["archives"]) == 3
    assert len(list((root / "assets").glob("*.usdz"))) == 3
    manifest = json.loads(
        next((root / "outputs").glob("*/*/scene-selection.json")).read_text()
    )
    assert manifest["revision"] == REVISION
    assert len(manifest["scenes"]) == 3
    assert "fake-test-token" not in json.dumps(manifest)
    assert not list((root / "assets").glob("*.part"))
    # Redirects to file servers must not carry the private HF credential.
    from urllib.request import Request

    request = Request(
        "https://huggingface.co/test", headers={"Authorization": "Bearer secret"}
    )
    redirected = namespace["SafeRedirect"]().redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example.org/file"
    )
    assert redirected.get_header("Authorization") is None
    with pytest.raises(RuntimeError, match="non-HTTPS"):
        namespace["SafeRedirect"]().redirect_request(
            request, None, 302, "Found", {}, "http://cdn.example.org/file"
        )


@pytest.mark.parametrize("condition", ["eligible", "oversized"])
def test_insufficient_or_oversized_candidates_download_nothing(download, condition):
    invoke, state, root = download
    state[condition] = 2 if condition == "eligible" else True
    with pytest.raises(AssertionError, match="Fewer than 3"):
        invoke()
    assert state["archives"] == []
    assert not (root / "assets").exists()


def test_bad_hash_never_publishes_scene(download):
    invoke, state, root = download
    state["corrupt"] = True
    with pytest.raises(AssertionError, match="verification failed"):
        invoke()
    assert len(state["archives"]) == 1
    assert not list((root / "assets").glob("*.usdz"))


@pytest.mark.parametrize("correct", [True, False])
def test_existing_scene_reused_or_preserved_without_overwrite(download, correct):
    invoke, state, root = download
    assets = root / "assets"
    assets.mkdir()
    content = b"fake scene data" if correct else b"different existing data"
    existing = assets / "0.usdz"
    existing.write_bytes(content)
    if correct:
        invoke()
        assert len(state["archives"]) == 2
    else:
        with pytest.raises(AssertionError, match="Existing file differs"):
            invoke()
        assert state["archives"] == []
    assert existing.read_bytes() == content
