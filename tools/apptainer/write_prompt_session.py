# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Copy a local video-model session request with a different positive prompt."""

import argparse
from pathlib import Path

from alpasim_grpc.v0 import video_model_pb2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    request = video_model_pb2.SessionRequest.FromString(args.source.read_bytes())
    request.text_prompt.positive = args.positive
    args.output.write_bytes(request.SerializeToString())
    print("Saved prompt-adjusted session request:", args.output, flush=True)


if __name__ == "__main__":
    main()
