from __future__ import annotations

import argparse
from pathlib import Path

from common.config_utils import base_model_from_file, base_model_to_file
from dl.submission import Submission as SubmissionDL
from postprocess.submission_utils import Submission as SubmissionGBDT


def _resolve(p: Path) -> Path:
    # strict=False avoids issues if something is temporarily missing; good for symlink targets.
    return p.resolve(strict=False)


def _materialize_dir_symlink(dst_path: Path) -> Path:
    """
    If dst_path is a symlink to a directory, replace it with a real directory and return
    the old symlink target directory path.
    """
    if not dst_path.is_symlink():
        return None

    target = dst_path.resolve(strict=False)
    if not target.is_dir():
        return None

    dst_path.unlink()  # remove symlink
    dst_path.mkdir()  # create real dir
    return target


def merge_symlinks(
    src_dir: Path, dst_dir: Path, *, skip_names: set[str] | None = None
) -> None:
    """
    Merge src_dir into dst_dir by creating symlinks.
    - Unique entries become symlinks (including directories).
    - If a directory name collides, merge contents recursively (src1 wins on file collisions).
    - If a file name collides, keep existing.
    """
    skip_names = skip_names or set()

    for entry in src_dir.iterdir():
        name = entry.name
        if name in skip_names:
            continue

        dst_entry = dst_dir / name

        # Case 1: dst doesn't exist -> just symlink (dir or file).
        if not (dst_entry.exists() or dst_entry.is_symlink()):
            dst_entry.symlink_to(_resolve(entry), target_is_directory=entry.is_dir())
            continue

        # Case 2: collision.
        # If both are dirs -> merge recursively.
        if entry.is_dir():
            # If dst is a symlink-to-dir, materialize it so we can merge inside.
            old_target = _materialize_dir_symlink(dst_entry)
            if old_target is not None:
                # First merge what dst previously pointed to, then merge current entry.
                merge_symlinks(old_target, dst_entry)
                merge_symlinks(entry, dst_entry)
                continue

            # If dst is already a real directory, merge into it.
            if dst_entry.is_dir():
                merge_symlinks(entry, dst_entry)
                continue

            # dst exists but is not a directory (file vs dir collision): keep existing.
            continue

        # entry is a file, dst exists already -> keep existing (src1 wins).
        continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src1",
        required=True,
        type=str,
        help="Path to dir with 1st submission to merge",
    )
    parser.add_argument(
        "--src2",
        required=True,
        type=str,
        help="Path to dir with 2nd submission to merge",
    )
    parser.add_argument(
        "--dst", required=True, type=str, help="Path to dir in which combine everything"
    )
    parser.add_argument(
        "--is-dl", dest="is_dl", action="store_true", help="Use DL Submission schema"
    )
    args = parser.parse_args()

    src1 = Path(args.src1)
    src2 = Path(args.src2)
    dst = Path(args.dst)

    if not src1.is_dir():
        raise NotADirectoryError(f"--src1 is not a directory: {src1}")
    if not src2.is_dir():
        raise NotADirectoryError(f"--src2 is not a directory: {src2}")

    # Create dst dir, raise if anything exists there.
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.mkdir(parents=True, exist_ok=False)

    submission_cls = SubmissionDL if args.is_dl else SubmissionGBDT

    submission_from1 = base_model_from_file(submission_cls, src1 / "submission.json")
    submission_from2 = base_model_from_file(submission_cls, src2 / "submission.json")

    submission_to = submission_cls()
    submission_to.models = list(submission_from1.models) + list(submission_from2.models)
    submission_to.thresholds = list(submission_from1.thresholds) + list(
        submission_from2.thresholds
    )

    base_model_to_file(submission_to, dst / "submission.json")

    # Merge directory trees; skip only root-level submission.json (we wrote merged one).
    skip = {"submission.json"}
    merge_symlinks(src1, dst, skip_names=skip)
    merge_symlinks(src2, dst, skip_names=skip)


if __name__ == "__main__":
    main()
