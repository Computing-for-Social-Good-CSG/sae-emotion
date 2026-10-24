#!/usr/bin/env python3
"""Offline structural checks; does not execute research code or download models."""
from pathlib import Path
import ast
import csv
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")
DATA_FILES = {
    "combined_emotions.csv": (",", ["emotion", "text", "dataset", "input_text"], 9870),
    "train_df.csv": (",", ["input_text", "emotion"], 2400),
    "val_df.csv": (",", ["input_text", "emotion"], 1974),
    "emoevent_en.csv": ("\t", ["id", "tweet", "emotion", "offensive"], 7303),
    "xed-en.csv": (",", ["text", "emotion"], 2803),
}

def bindings(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split('.')[0] for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return names

def main():
    failures = []
    counts = {"python_files": 0, "notebooks": 0, "code_cells": 0, "data_files": 0}
    active_imports = []
    module_bindings = {}
    for p in (ROOT / 'src' / 'sae_emotion').glob('*.py'):
        module_bindings['sae_emotion.' + p.stem] = bindings(ast.parse(p.read_text()))
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or any(part in {'.git', '.venv', '__pycache__'} for part in p.parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix in {'.py', '.ipynb', '.md', '.json', '.toml', '.sbatch', '.csv'}:
            text = p.read_text(encoding='utf-8')
            if TOKEN_PATTERN.search(text):
                failures.append(f'{rel}: credential-like token detected (value withheld)')
        if p.suffix == '.py':
            counts['python_files'] += 1
            sources = [(None, p.read_text())]
        elif p.suffix == '.ipynb':
            counts['notebooks'] += 1
            try:
                nb = json.loads(p.read_text())
            except ValueError:
                failures.append(f'{rel}: invalid JSON')
                continue
            if nb.get('nbformat') != 4:
                failures.append(f'{rel}: expected notebook format 4')
            ids = [c.get('id') for c in nb['cells']]
            if len(ids) != len(set(ids)) or None in ids:
                failures.append(f'{rel}: missing or duplicate cell IDs')
            sources = []
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    counts['code_cells'] += 1
                    if cell.get('outputs') or cell.get('execution_count') is not None:
                        failures.append(f'{rel} cell {i}: saved execution artifacts remain')
                    sources.append((i, ''.join(cell.get('source', []))))
        else:
            continue
        for index, source in sources:
            location = rel if index is None else f'{rel} cell {index}'
            try:
                tree = ast.parse(source)
                compile(tree, location, 'exec')
            except SyntaxError as exc:
                failures.append(f'{location}:{exc.lineno}: {exc.msg}')
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('sae_emotion.'):
                    active_imports.append((location, node))
    for location, node in active_imports:
        if node.module not in module_bindings:
            failures.append(f'{location}: package module absent: {node.module}')
            continue
        for alias in node.names:
            if alias.name != '*' and alias.name not in module_bindings[node.module]:
                failures.append(f'{location}: cannot locate {alias.name} in {node.module}')
    for filename, (delimiter, columns, expected_rows) in DATA_FILES.items():
        p = ROOT / 'data' / filename
        if not p.is_file():
            failures.append(f'data/{filename}: required source CSV is missing')
            continue
        counts['data_files'] += 1
        with p.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = sum(1 for _ in reader)
            if reader.fieldnames != columns or rows != expected_rows:
                failures.append(f'data/{filename}: unexpected columns or row count')
    result = {
        'status': 'passed' if not failures else 'failed',
        'scope': 'Offline syntax, notebook cleanup, credential-pattern scan, package imports, and source CSV columns/counts.',
        'counts': counts,
        'failures': failures,
        'not_tested': ['GPU execution', 'third-party dependency compatibility', 'model/SAE downloads', 'scientific equivalence or reproduction of paper results'],
    }
    print(json.dumps(result, indent=2))
    return 1 if failures else 0

if __name__ == '__main__':
    sys.exit(main())
