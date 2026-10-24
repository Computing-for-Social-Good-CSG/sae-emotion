# Recorded cluster environment

The author supplied `environment/environment_report.json` and `environment/package_versions.txt`. They agree on all **245 installed package versions**. The recorded interpreter is `/projects/CSGLab/shu.b/envs/sae-lens/bin/python`, on Linux x86_64.

| Component | Recorded version |
| --- | --- |
| GPU | NVIDIA H200 (author reported) |
| Python | 3.11.14 |
| PyTorch | 2.13.0+cu126 |
| CUDA used to build PyTorch | 12.6 |
| cuDNN reported by PyTorch | 91002 |
| Transformers | 4.57.6 |
| SAELens | 6.37.3 |
| NNsight | 0.5.15 |
| Accelerate | 1.14.0 |
| NumPy | 2.4.6 |
| pandas | 3.0.5 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| Hugging Face Hub | 0.36.2 |
| safetensors | 0.8.0 |

The author confirmed **NVIDIA H200** as the experiment GPU; this is recorded separately in `environment/hardware.json`. The original capture reported `cuda_available: false` and no `nvidia-smi`, so it is preserved unchanged. That can occur on a login node or outside a GPU allocation and does not by itself establish that the CUDA-enabled PyTorch installation is broken. The NVIDIA driver version remains unspecified; GPU count, memory capacity, and form factor have not been inferred.

## Dependency files and setup

- `environment/environment_report.json`: original uploaded report, unchanged.
- `environment/hardware.json`: author-reported NVIDIA H200 GPU; driver version remains unspecified.
- `environment/package_versions.txt`: original full version inventory, unchanged.
- `requirements/research-cluster.txt`: exact versions for the 18 direct research dependencies.
- `requirements/notebooks-cluster.txt`: recorded JupyterLab and kernel versions.
- `requirements/lora-cluster.txt`: recorded PEFT version for the optional historical LoRA baseline.
- `results/validation/environment_snapshot.json`: consistency check of the two uploads and dependency coverage.

To use the existing cluster environment, call its recorded interpreter directly and register the repository without upgrading dependencies:

```bash
/projects/CSGLab/shu.b/envs/sae-lens/bin/python -m pip install --no-deps --no-build-isolation -e .
```

Run the installation command from the repository root. Before launching later commands, activate that environment using your usual cluster setup, or continue using the full interpreter path.

For a new environment, target Python 3.11.14 and use the recorded dependency pins rather than unconstrained newest packages. The exact package indexes/wheel origins were not exported; the `torch==2.13.0+cu126` build needs a source that actually provides that build. An installation from these pins has **not** been independently tested. The full 245-package list describes the author's environment; it may include tools unnecessary for running this repository.

The exports record the environment at capture time. They do not establish whether it changed after the final paper runs. The local lexical-preprocessing check used a different, explicitly recorded Mac environment; it is not a validation of these cluster dependencies.

## Refresh or collect a report

Run these checks **on the machine and in the Python environment that ran the experiments**. A Mac export describes your Mac, not the cluster environment. If the environment has changed since the experiments, label the export with that fact.

## From the experiment notebook

Open a notebook with the same kernel used for the experiments and run:

```python
import sys
import importlib.metadata as metadata

print("Python executable:", sys.executable)
print("Python version:", sys.version)
for name in ["torch", "transformers", "sae-lens", "nnsight", "numpy", "pandas",
             "scikit-learn", "scipy", "datasets", "huggingface-hub", "safetensors"]:
    try:
        print(name, metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name, "not installed")
```

The executable path identifies the actual notebook environment. Terminal `python` may point to a different interpreter.

## Save a shareable report

Copy this repository to the experiment machine, open a terminal in its root, activate the original environment, and run:

```bash
python scripts/collect_environment.py
```

It creates two small files:

- `environment/environment_report.json`: Python, package versions, platform, and PyTorch/CUDA/GPU information when available.
- `environment/package_versions.txt`: installed package names and versions.

To use the notebook's exact interpreter, run this from a notebook started at the repository root:

```python
import sys
import subprocess

subprocess.run([sys.executable, "scripts/collect_environment.py"], check=True)
```

The collector does not install packages, download models, dump environment variables, or include private package URLs. It only writes version information. It has a main guard, so importing it does not write files.

Run it on a GPU node to capture the driver and GPU type. A report from a login node can still capture package versions; it may correctly report that CUDA is unavailable there. The installed NVIDIA driver and the CUDA version used to build PyTorch are different pieces of information, so both are recorded when possible.

## Quick terminal alternative

In the activated experiment environment:

```bash
python --version
python -m pip freeze > requirements-original.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
nvidia-smi
```

Use `python -m pip` so the package list comes from the same interpreter. `pip freeze` may also include local/editable install paths or package-source URLs; the repository collector provides a version-only export if those are not appropriate to share.

Keep the committed report and dependency pins in sync if a new environment is deliberately adopted. A newly collected report cannot recover earlier package versions if the environment was upgraded after the experiments.
