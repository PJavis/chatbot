#!/usr/bin/env python3
"""Ensure the trained model archive contains flow-retrieval FAISS artifacts."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path


FLOW_RESOURCE = "components/train_SearchReadyLLMCommandGenerator0"
REQUIRED_FILES = ("index.faiss", "index.pkl")


def model_has_flow_index(model_path: Path) -> bool:
    with tarfile.open(model_path, "r:gz") as archive:
        names = set(archive.getnames())
    return all(f"{FLOW_RESOURCE}/{name}" in names for name in REQUIRED_FILES)


def find_cache_artifact(cache_dir: Path) -> Path | None:
    candidates = []
    for path in cache_dir.glob("tmp*/flow_retrieval_config.json"):
        candidate_dir = path.parent
        if all((candidate_dir / name).exists() for name in REQUIRED_FILES):
            candidates.append(candidate_dir)

    if not candidates:
        return None

    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def inject_flow_index(model_path: Path, artifact_dir: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=model_path.parent, prefix=f"{model_path.stem}-", suffix=".tar.gz", delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        with tarfile.open(model_path, "r:gz") as source, tarfile.open(tmp_path, "w:gz") as target:
            for member in source.getmembers():
                extracted = source.extractfile(member) if member.isfile() else None
                target.addfile(member, extracted)

            for name in REQUIRED_FILES:
                file_path = artifact_dir / name
                arcname = f"{FLOW_RESOURCE}/{name}"
                target.add(file_path, arcname=arcname)

        shutil.move(tmp_path, model_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch a Rasa model archive with flow-retrieval FAISS artifacts."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to the model .tar.gz file")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".rasa/cache"),
        help="Path to the Rasa cache directory containing flow-retrieval artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    cache_dir = args.cache_dir.resolve()

    if not model_path.exists():
        raise SystemExit(f"Model archive not found: {model_path}")

    if model_has_flow_index(model_path):
        print(f"Model already contains flow-retrieval FAISS files: {model_path}")
        return 0

    artifact_dir = find_cache_artifact(cache_dir)
    if artifact_dir is None:
        raise SystemExit(
            "Could not find cached flow-retrieval artifacts with both index.faiss and index.pkl"
        )

    inject_flow_index(model_path, artifact_dir)
    print(f"Injected flow-retrieval FAISS files from {artifact_dir} into {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
