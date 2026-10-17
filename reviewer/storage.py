"""Inspectable result JSON in exclusive run directories; never overwrite a run."""

import json
from pathlib import Path
import re
import shutil
import uuid

from reviewer.schemas import BaselineRun


def save_run(run: BaselineRun, directory: str | Path, *, run_id: str | None = None) -> Path:
    # Validate at the persistence boundary, including nested changes after creation.
    run = BaselineRun.model_validate(run.model_dump())
    run_id = uuid.uuid4().hex if run_id is None else run_id
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("run_id must be a simple filename component")
    text = json.dumps(run.model_dump(mode="json"), indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False) + "\n"
    parent = Path(directory)
    if any(path.is_symlink() for path in (parent, *parent.parents)):
        raise ValueError("results directory must not contain symlink ancestors")
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / run_id
    target.mkdir(exist_ok=False)  # Collision is an error, never overwrite or merge.
    try:
        path = target / (run.result.mode + '.json')
        temporary = target / ('.' + run.result.mode + '.json.tmp')
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
        temporary.rename(path)
        return path
    except BaseException:
        shutil.rmtree(target)
        raise
