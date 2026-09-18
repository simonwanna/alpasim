# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Inspect recorded map/trajectory context without starting simulator services."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {}
    with ZipFile(args.scene) as archive:

        def read_member(name):
            if archive.getinfo(name).file_size > 64 * 1024 * 1024:
                raise ValueError(f"Inspection member exceeds 64 MiB: {name}")
            return archive.read(name)

        fig, ax = plt.subplots(figsize=(12, 12))
        ego_xy = None
        # Same ClipGT geometry fields as src/tools/map_utils/plot_map.py.
        for kind in (
            "lane",
            "wait_line",
            "crosswalk",
            "egomotion_estimate",
            "intersection_area",
        ):
            name = f"clipgt/{kind}.parquet"
            if name not in archive.namelist():
                report[kind] = {"present": False}
                continue
            table = pq.ParquetFile(pa.BufferReader(read_member(name))).read(
                use_threads=False
            )
            report[kind] = {
                "schema": str(table.schema),
                "rows": table.num_rows,
                "sample": table.slice(0, 1).to_pylist(),
            }
            if kind == "intersection_area":
                continue  # Retain schema evidence; do not guess its geometry fields.
            elements = table[kind].to_pylist()
            if not elements:
                continue
            if kind == "egomotion_estimate":
                report["map_route_samples"] = table.take(
                    pa.array(range(0, table.num_rows, max(1, table.num_rows // 100)))
                ).to_pylist()
                ego_xy = np.asarray(
                    [[e["location"]["x"], e["location"]["y"]] for e in elements]
                )
                ax.plot(
                    *ego_xy.T,
                    color="black",
                    linewidth=2,
                    label="Recorded ego (map coordinates)",
                )
                ax.scatter(*ego_xy[0], color="blue", s=60, label="Recording start")
                continue
            for element in elements:
                fields = (
                    ("left_rail", "right_rail") if kind == "lane" else ("location",)
                )
                for field in fields:
                    points = np.asarray([[p["x"], p["y"]] for p in element[field]])
                    if len(points) < 2:
                        continue
                    color = {"lane": "gray", "wait_line": "red", "crosswalk": "green"}[
                        kind
                    ]
                    ax.plot(*points.T, color=color, linewidth=0.7, alpha=0.8)
        ax.set_aspect("equal")
        ax.set_title("Recorded map: gray lanes, red wait lines, green crosswalks")
        if ego_xy is not None:
            lower, upper = ego_xy.min(axis=0) - 50, ego_xy.max(axis=0) + 50
            ax.set_xlim(lower[0], upper[0])
            ax.set_ylim(lower[1], upper[1])
            ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(args.output_dir / "map-overview.png", dpi=140)
        plt.close(fig)

        raw = json.loads(read_member("rig_trajectories.json"))
        report["coordinate_metadata"] = {
            key: raw[key] for key in ("world_to_nre", "T_world_base") if key in raw
        }
        (rig,) = raw["rig_trajectories"]
        times = np.asarray(rig["T_rig_world_timestamps_us"], dtype=np.int64)
        matrices = np.asarray(rig["T_rig_worlds"], dtype=float)
        if (
            matrices.shape != (len(times), 4, 4)
            or len(times) < 2
            or np.any(np.diff(times) <= 0)
            or not np.isfinite(matrices).all()
        ):
            raise ValueError("Invalid recorded rig poses/timestamps")
        seconds = (times - times[0]) / 1e6
        yaw = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
        xy = matrices[:, :2, 3]
        report["rig"] = {
            "first_timestamp_us": int(times[0]),
            "last_timestamp_us": int(times[-1]),
            "first_xy": xy[0].tolist(),
            "last_xy": xy[-1].tolist(),
        }
        report["seed_images"] = [
            n
            for n in archive.namelist()
            if n.startswith("frames/") and n.endswith((".jpg", ".jpeg", ".png"))
        ]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].plot(*xy.T)
        for sec in range(0, int(seconds[-1]) + 1, 5):
            i = int(np.searchsorted(seconds, sec))
            axes[0].annotate(f"{sec}s", xy[i])
        axes[0].set_aspect("equal")
        axes[0].set_title("Recorded rig path (native rig coordinates)")
        axes[1].plot(seconds, np.rad2deg(yaw - yaw[0]))
        axes[1].set_xlabel("Seconds from first rig pose")
        axes[1].set_ylabel("Recorded heading change (degrees)")
        axes[1].set_title("Heading change is not proof of an intersection")
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.savefig(args.output_dir / "recorded-route.png", dpi=140)
        plt.close(fig)
        report["rig_samples"] = [
            {
                "timestamp_us": int(times[i]),
                "elapsed_s": float(seconds[i]),
                "x": float(xy[i, 0]),
                "y": float(xy[i, 1]),
                "heading_delta_deg": float(np.rad2deg(yaw[i] - yaw[0])),
            }
            for i in range(0, len(times), max(1, len(times) // 100))
        ]
    (args.output_dir / "scene-inspection.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    for kind in ("intersection_area", "egomotion_estimate"):
        print(kind, json.dumps(report[kind], default=str)[:3000], flush=True)
    print("Map:", args.output_dir / "map-overview.png", flush=True)
    print("Recorded route:", args.output_dir / "recorded-route.png", flush=True)
    print(
        "Inspection only: intersection approach still requires verification", flush=True
    )


if __name__ == "__main__":
    main()
