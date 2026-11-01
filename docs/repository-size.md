# Repository size and artifact policy

The 2026-11-01 cleanup removes large, non-runtime artifacts from the current
checkout. It preserves Git history and all training and inference source code.
The reference version before cleanup is
`3b8872bd93d140dde6e7f95fdd996edabb6b75c9`.

## Removed artifacts

| Category | Files | Size before removal |
| --- | ---: | ---: |
| Evaluation videos under `evaluation/` and `evalution/` | 620 | 165.56 MiB |
| Generated HTML, Plotly bundle, charts, CSV and temporary Zarr data under `examples/umi/analyze/` | 25 | 28.86 MiB |
| ALOHA CAD, STL, ZIP and assembly PDF under `third_party/aloha/aloha2/` | 12 | 97.51 MiB |
| Unreferenced `docs/Past-Future Temporal Bottleneck for Memory-Augmented Future-Aware Vision-Language-Action Control_v3.pdf` | 1 | 6.60 MiB |
| Root debug plots, Finder metadata and an editor swap file | 4 | 0.20 MiB |

In total, 662 files (298.74 MiB) are removed from a 356.78 MiB tracked tree,
leaving approximately 58.05 MiB of source, dependencies and retained records
after adding the cleanup documentation and size check, an 83.7% reduction.
These figures exclude the separate Git object database.

The JSON/JSONL evaluation metrics and labels, evaluation README, and
`ablation_results/` remain available for the existing experiment reports and
summary scripts. All analysis Python and shell scripts remain available; they
can generate fresh reports from the original datasets. LeRobot code, its test
fixtures and documentation media, LIBERO runtime assets and notebook examples,
and ALOHA runtime scripts and ROS configuration are retained.

The removed ALOHA files are mechanical manufacturing and assembly resources;
the ALOHA example's Docker build and runtime use the retained source and
configuration. For the exact historical hardware files, recover them from the
reference commit below. The vendored ALOHA README also points to its upstream
hardware tutorials.

## Recover a historical file

Recover only the needed file, for example:

```bash
git show '3b8872bd93d140dde6e7f95fdd996edabb6b75c9:docs/Past-Future Temporal Bottleneck for Memory-Augmented Future-Aware Vision-Language-Action Control_v3.pdf' > /tmp/openpi-memory-paper.pdf
```

In a partial or shallow clone this may require fetching the historical commit
or blob. Removing files from the current version does not reclaim historical
Git objects or reduce the size of a full-history clone. A shallow clone of the
published cleaned version can avoid downloading that older history.

## Keep generated outputs outside source control

Store datasets, weights, evaluation videos, generated reports and hardware
design downloads outside the repository, or in the existing ignored output
directories. Small experiment summaries and required test fixtures can remain
tracked. Ignore rules prevent the removed analysis exports, Finder metadata
and editor swap files from being added again by normal `git add` usage.

Check the current tracked working-tree files without training dependencies:

```bash
python3 scripts/check_repo_size.py
```

Before committing, check the actual staged blobs:

```bash
python3 scripts/check_repo_size.py --staged
```

The local pre-commit hook uses the staged check to reject files larger than
5 MiB. It runs when pre-commit is installed and enabled; the check can also be
run directly. Keep large artifacts in external storage rather than raising
the limit for generated outputs.
