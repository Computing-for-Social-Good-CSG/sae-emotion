# Recorded cluster dependency versions

These pins are derived from the uploaded environment inventory; they are not guessed version bounds and have not been installed or GPU-tested by the organizing pass.

`research-cluster.txt` includes the 18 direct research dependencies from `pyproject.toml`. `notebooks-cluster.txt` and `lora-cluster.txt` capture optional tools. The original complete inventory is preserved in `../environment/package_versions.txt`; the platform, interpreter, and CUDA-build information are in `../environment/environment_report.json`.

Use Python 3.11.14 for matching the snapshot. Prefer the existing cluster environment and install this repository with `--no-deps --no-build-isolation -e .` to avoid changing its packages. Reconstructing a fresh environment also requires a source for the exact `torch==2.13.0+cu126` build; package indexes/wheel origins are not present in the export. The full inventory includes unrelated tools and platform-specific dependencies, so it is not presented as a universal Mac/Linux installation command.

The generic `pyproject.toml` extras remain useful for identifying required libraries. They do not override the recorded version files or certify compatibility with arbitrary newer releases.
