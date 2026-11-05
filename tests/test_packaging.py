import hashlib
import io
import json
import zipfile
import pytest
from scripts.package_heatline import included, write_code
from scripts.package_submission import package_submission


@pytest.mark.parametrize('name', ['frontend/node_modules/package/index.js', 'frontend/.next/server.js',
    'frontend/.env.local', '.git/config', 'frontend/AGENTS.md', 'PROJECT_GUIDE.md',
    'frontend/tsconfig.tsbuildinfo', 'src/__pycache__/module.pyc', 'data/../../secret.txt'])
def test_package_excludes_local_state(name):
    assert not included(name)


def test_package_keeps_runtime_examples_and_checksums(tmp_path):
    (tmp_path / 'frontend').mkdir()
    (tmp_path / 'frontend/.env.example').write_text('NEXT_PUBLIC_API_URL=')
    (tmp_path / 'frontend/.env.local').write_text('SECRET=must-not-ship')
    (tmp_path / 'README.md').write_text('Run the project')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        write_code(archive, tmp_path, 'MiniProject123/Code')
    with zipfile.ZipFile(stream) as archive:
        manifest = json.loads(archive.read('MiniProject123/Code/SHA256SUMS.json'))
        assert 'frontend/.env.example' in manifest
        assert 'frontend/.env.local' not in manifest
        assert manifest['README.md'] == hashlib.sha256(b'Run the project').hexdigest()


def test_invalid_roll_cannot_change_package_path(tmp_path):
    with pytest.raises(ValueError, match='digits'):
        package_submission('../outside', tmp_path, tmp_path / 'output.zip')
