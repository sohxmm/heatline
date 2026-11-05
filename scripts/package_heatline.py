"""Build a reproducible source archive without local state or credentials."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('src/', 'api/', 'tests/', 'docs/', 'data/', 'artifacts/', 'models/', 'frontend/')
FILES = {'README.md', 'config.yaml', 'config_v2.yaml', 'render.yaml', 'requirements.txt',
         'requirements-api.txt', 'requirements-lock.txt', 'requirements-app.txt',
         'notebooks/03_heatline_lab.ipynb', 'results/tables/baseline_comparison.csv',
         'legacy/common-input-study-v1.zip', 'legacy/README.md'}
SCRIPTS = {'package_heatline.py', 'package_submission.py', 'build_lab_notebook.py',
           'export_evidence.py', 'export_seasonal_evidence.py', 'export_metric_context.py',
           'fetch_weather_snapshot.py', 'audit_station_sources.py'}
EXCLUDED_PARTS = {'.git', '.venv', 'venv', 'node_modules', '.next', '.vercel', '__pycache__',
                  '.pytest_cache', '.ipynb_checkpoints', 'test-results', 'playwright-report',
                  'seasonal-study', 'dist'}


def included(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or set(path.parts) & EXCLUDED_PARTS:
        return False
    if path.name in {'AGENTS.md', 'PROJECT_GUIDE.md'} or path.suffix in {'.pyc', '.log', '.tsbuildinfo'}:
        return False
    if path.name.startswith('.env') and path.name != '.env.example':
        return False
    return (name in FILES or name.startswith(DIRECTORIES)
            or (path.parent == PurePosixPath('scripts') and path.name in SCRIPTS))


def source_paths(root=ROOT):
    root = Path(root).resolve()
    if (root / '.git').exists():
        names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    else:
        names = (p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
    paths = []
    for name in sorted(set(filter(None, names))):
        path = root / name
        if included(name) and path.is_file():
            if not path.resolve().is_relative_to(root):
                raise ValueError(f'Source link leaves project directory: {name}')
            paths.append(name)
    return paths


def write_code(archive, root=ROOT, prefix='heatline'):
    root = Path(root)
    files = source_paths(root)
    hashes = {}
    for name in files:
        data = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        archive.writestr(f'{prefix}/{name}', data)
    archive.writestr(f'{prefix}/SHA256SUMS.json', json.dumps(hashes, indent=2))
    return files


def package(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        files = write_code(archive)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity check failed')
    print(f'{destination}: {len(files)} files; {destination.stat().st_size:,} bytes')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'dist/heatline-submission.zip'))
    package(parser.parse_args().output)
