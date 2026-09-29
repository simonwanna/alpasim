# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Read scene geometry and print bounded intersection candidates; no model load."""
import json
import sys
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq


def inside(points, polygon):
    result = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if abs(b[1] - a[1]) < 1e-12:
            continue
        crosses = (a[1] > y) != (b[1] > y)
        edge_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        result ^= crosses & (x < edge_x)
    return result


def xy(points):
    return np.array([[p['x'], p['y']] for p in points], dtype=float)


def inspect(scene):
    with ZipFile(scene) as archive:
        rigs = json.loads(archive.read('rig_trajectories.json'))['rig_trajectories']
        assert len(rigs) == 1, 'Expected one recorded rig'
        rig = rigs[0]
        poses = np.asarray(rig['T_rig_worlds'], dtype=float)
        times = np.asarray(rig['T_rig_world_timestamps_us'], dtype=np.int64)
        assert poses.shape == (len(times), 4, 4)
        points = poses[:, :2, 3]
        tables = {name: pq.read_table(BytesIO(archive.read(f'clipgt/{name}.parquet'))).to_pylist()
                  for name in ('lane', 'intersection_area')}
    print('Recorded start XY:', points[0].round(2).tolist())
    candidates = []
    for row in tables['intersection_area']:
        polygon = xy(row['intersection_area']['location'])
        if len(polygon) < 3:
            continue
        hits = np.flatnonzero(inside(points, polygon))
        if len(hits):
            candidates.append((int(hits[0]), str(row['key']['map_id']), row))
    candidates.sort(key=lambda item: (item[0], item[1]))
    print('Intersected map polygons:', len(candidates))
    for index, ident, row in candidates[:3]:
        print('CANDIDATE', ident, row['intersection_area']['category'],
              'first sample seconds from start:', round((times[index]-times[0])/1e6, 2),
              'XY:', points[index].round(2).tolist(), 'contains start:', index == 0)
    future = [c for c in candidates if c[0] > 0]
    if not future:
        print('No later intersection found; do not invent a turn route.')
        return
    index, ident, row = future[0]
    print('Inspecting earliest later polygon:', ident)
    nearby = []
    for lane_row in tables['lane']:
        lane = lane_row['lane']
        left, right = xy(lane['left_rail']), xy(lane['right_rail'])
        if len(left) < 2 or len(right) < 2:
            continue
        start, end = (left[0]+right[0])/2, (left[-1]+right[-1])/2
        vector = end-start
        length2 = float(vector @ vector)
        if length2 < 1e-8:
            continue
        fraction = np.clip((points[index]-start) @ vector / length2, 0, 1)
        distance = float(np.linalg.norm(points[index]-(start+fraction*vector)))
        heading = np.arctan2(vector[1], vector[0])-np.arctan2(poses[index, 1, 0], poses[index, 0, 0])
        angle = float((np.degrees(heading)+180) % 360-180)
        nearby.append((distance, str(lane_row['key']['map_id']), lane['lane_direction'],
                       round(angle, 1), start.round(2).tolist(), end.round(2).tolist()))
    for distance, ident, direction, angle, start, end in sorted(nearby)[:10]:
        print('LANE', ident, direction, 'chord distance m:', round(distance, 1),
              'heading relative to ego deg:', angle, 'start/end:', start, end)
    print('Candidates only: sampled polygon entry and endpoint chords, not a validated connected route.')


if __name__ == '__main__':
    inspect(sys.argv[1])
