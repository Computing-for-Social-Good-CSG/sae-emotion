#!/usr/bin/env python3
"""Record versions from the Python interpreter running this script; no network calls."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys


def collect():
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get('Name')
        if name:
            packages[name] = dist.version
    packages = dict(sorted(packages.items(), key=lambda item: item[0].lower()))
    report = {
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'operating_system': platform.system(),
        'os_release': platform.release(),
        'machine_architecture': platform.machine(),
        'packages': packages,
        'notes': [
            'This reports the current interpreter, not a historical environment.',
            'Package names and versions are recorded without credentials, environment variables, or private installation URLs.',
            'This is a version inventory, not a portable lockfile; editable/local builds and CUDA wheels may need additional provenance.',
        ],
    }
    try:
        import torch
        available = torch.cuda.is_available()
        report['pytorch'] = {
            'version': str(torch.__version__),
            'built_with_cuda': torch.version.cuda,
            'cuda_available': available,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else [],
        }
    except ImportError:
        report['pytorch'] = {'status': 'not installed in this interpreter'}
    except Exception as exc:
        report['pytorch'] = {'status': 'import/device inspection failed', 'error_type': type(exc).__name__}
    executable = shutil.which('nvidia-smi')
    if executable:
        try:
            proc = subprocess.run(
                [executable, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=20,
            )
            report['nvidia_driver'] = {'exit_code': proc.returncode, 'gpu_summary': proc.stdout.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            report['nvidia_driver'] = {'status': type(exc).__name__}
    else:
        report['nvidia_driver'] = {'status': 'nvidia-smi unavailable; run on a GPU node for driver information'}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('environment'))
    args = parser.parse_args(argv)
    report = collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / 'environment_report.json'
    version_path = args.output_dir / 'package_versions.txt'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    version_path.write_text(''.join(f'{name}=={version}\n' for name, version in report['packages'].items()))
    print(f'Python interpreter: {sys.executable}')
    print(f'Python version: {report["python_version"]}')
    print(f'Wrote {report_path.resolve()}')
    print(f'Wrote {version_path.resolve()}')


if __name__ == '__main__':
    main()
