"""Package the assessed source, cached data and evidence, excluding local tooling."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES=('src/','api/','tests/','docs/','data/raw/','data/processed/','artifacts/heatline/','legacy/','frontend/')
FILES={'README.md','PROJECT_GUIDE.md','config.yaml','render.yaml','requirements.txt','requirements-api.txt','requirements-lock.txt',
    'notebooks/03_heatline_lab.ipynb','scripts/build_lab_notebook.py','scripts/export_evidence.py','scripts/fetch_weather_snapshot.py','scripts/package_heatline.py'}


def package(destination):
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    files=[p for p in tracked if p and (p in FILES or p.startswith(DIRECTORIES))]
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files: archive.write(ROOT/p,f'heatline/{p}')
        archive.writestr('heatline/SHA256SUMS.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert 'heatline/frontend/package.json' in archive.namelist()
        assert 'heatline/artifacts/heatline/metrics.csv' in archive.namelist()
        assert not any('/node_modules/' in name or '/.env.local' in name for name in archive.namelist())
    print(f'{destination}: {len(files)} source/data files; {destination.stat().st_size:,} bytes; SHA-256 manifest included')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'dist/heatline-submission.zip'))
    package(parser.parse_args().output)
