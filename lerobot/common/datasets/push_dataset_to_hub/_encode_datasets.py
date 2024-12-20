#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Use this script to batch encode lerobot dataset from their raw format to LeRobotDataset and push their updated
version to the hub. Under the hood, this script reuses 'push_dataset_to_hub.py'. It assumes that you already
downloaded raw datasets, which you can do with the related '_download_raw.py' script.

For instance, for codebase_version = 'v1.6', the following command was run, assuming raw datasets from
lerobot-raw were downloaded in 'raw/datasets/directory':
```bash
python lerobot/common/datasets/push_dataset_to_hub/_encode_datasets.py \
  --raw-dir raw/datasets/directory \
  --raw-repo-ids lerobot-raw \
  --local-dir push/datasets/directory \
  --tests-data-dir tests/data \
  --push-repo lerobot \
  --vcodec libsvtav1 \
  --pix-fmt yuv420p \
  --g 2 \
  --crf 30
```
"""

from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub._download_raw import AVAILABLE_RAW_REPO_IDS
from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
from lerobot.scripts.push_dataset_to_hub import push_dataset_to_hub


def get_push_repo_id_from_raw(raw_repo_id: str, push_repo: str) -> str:
    dataset_id_raw = raw_repo_id.split("/")[1]
    dataset_id = dataset_id_raw.removesuffix("_raw")
    return f"{push_repo}/{dataset_id}"


def encode_datasets(
    raw_dir: Path,
    raw_repo_ids: list[str],
    push_repo: str,
    vcodec: str,
    pix_fmt: str,
    g: int,
    crf: int,
    local_dir: Path | None = None,
    tests_data_dir: Path | None = None,
    raw_format: str | None = None,
    dry_run: bool = False,
) -> None:
    if len(raw_repo_ids) == 1 and raw_repo_ids[0].lower() == "lerobot-raw":
        raw_repo_ids_format = AVAILABLE_RAW_REPO_IDS
    else:
        if raw_format is None:
            raise ValueError(raw_format)
        raw_repo_ids_format = {id_: raw_format for id_ in raw_repo_ids}

    for raw_repo_id, repo_raw_format in raw_repo_ids_format.items():
        check_repo_id(raw_repo_id)
        dataset_repo_id_push = get_push_repo_id_from_raw(raw_repo_id, push_repo)
        dataset_raw_dir = raw_dir / raw_repo_id
        dataset_dir = local_dir / dataset_repo_id_push if local_dir is not None else None
        encoding = {
            "vcodec": vcodec,
            "pix_fmt": pix_fmt,
            "g": g,
            "crf": crf,
        }

        if not (dataset_raw_dir).is_dir():
            raise NotADirectoryError(dataset_raw_dir)

        if not dry_run:
            push_dataset_to_hub(
                dataset_raw_dir,
                raw_format=repo_raw_format,
                repo_id=dataset_repo_id_push,
                local_dir=dataset_dir,
                resume=True,
                encoding=encoding,
                tests_data_dir=tests_data_dir,
            )
        else:
            print(
                f"DRY RUN: {dataset_raw_dir}  -->  {dataset_dir}  -->  {dataset_repo_id_push}@{CODEBASE_VERSION}"
            )

