"""Preserve only assigned reflective-camera SDK changes after four pinned patches."""
import difflib
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sdk = root.parent/'Weld-seam-finder'
    revision = (root/'WELDFINDER_REVISION').read_text().strip()
    names = ['weldfinder-continuous-scanning.patch', 'weldfinder-seam-api.patch',
             'weldfinder-coped-joint.patch', 'weldfinder-laser-preview-overlay.patch',
             'weldfinder-reflective-profiles.patch']
    files = [Path('app')/name for name in (
        'weldfinder/calibration_camera.py', 'weldfinder/pipeline.py', 'weldfinder/config.py',
        'weldfinder/detection.py', 'weldfinder/exposure.py', 'weldfinder/pointcloud.py',
        'weldfinder/seam.py', 'weldfinder/geometry.py', 'weldfinder/__init__.py',
        'tests/test_reflective_context.py', 'tests/test_native_laser_snapshot.py',
        'tests/test_reflective_processing.py', 'tests/reflective_native_island.json',
        'tests/test_pipeline_api.py', 'SDK.md')]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes = {str(path): sha(sdk/path) for path in files}
    archive = subprocess.check_output(['git', 'archive', '--format=tar', revision, 'app'], cwd=sdk)

    def unpack(path):
        path.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
            tree.extractall(path, filter='data')

    def apply(path, name):
        patch = root/'patches'/name
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=path, check=True, capture_output=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=path, check=True, capture_output=True)

    with tempfile.TemporaryDirectory(prefix='reflective-sdk-reproduction-') as temporary:
        temporary = Path(temporary)
        base, desired, fresh = (temporary/name for name in ('base', 'desired', 'fresh'))
        unpack(base)
        for name in names[:-1]:
            apply(base, name)
        shutil.copytree(base, desired)
        sections = []
        for path in files:
            old, new = base/path, desired/path
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sdk/path, new)
            before = old.read_text().splitlines(keepends=True) if old.exists() else []
            after = new.read_text().splitlines(keepends=True)
            if before == after:
                continue
            assert all(not lines or lines[-1].endswith('\n') for lines in (before, after)), str(path)
            name = str(path)
            header = f'diff --git a/{name} b/{name}\n'
            if not old.exists():
                header += 'new file mode 100644\n'
            sections.append(header+''.join(difflib.unified_diff(before, after,
                fromfile=f'a/{name}' if old.exists() else '/dev/null', tofile=f'b/{name}', n=3)))
        patch = root/'patches'/names[-1]
        patch.write_text(''.join(sections))
        unpack(fresh)
        for name in names:
            apply(fresh, name)
        reproduced = {str(path): sha(fresh/path) for path in files}
        assert reproduced == source_hashes
        expected_tree = {str(path.relative_to(desired)): sha(path) for path in desired.rglob('*') if path.is_file()}
        actual_tree = {str(path.relative_to(fresh)): sha(path) for path in fresh.rglob('*') if path.is_file()}
        assert actual_tree == expected_tree
        assert source_hashes == {str(path): sha(sdk/path) for path in files}, 'SDK changed while building artifact'
        result = dict(passed=True, base_revision=revision, patches_in_order=names,
            patch_sha256={name: sha(root/'patches'/name) for name in names},
            selected_paths=[str(path) for path in files], reproduced_sha256=reproduced,
            complete_reproduction_tree_matches=True, unrelated_worktree_changes_included=False,
            source_worktree_mutated=False)
        evidence = root/'evidence/reflective-tubes-20261030'
        evidence.mkdir(exist_ok=True, parents=True)
        (evidence/'sdk-patch-reproduction.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(passed=True, paths=len(files), patch_bytes=patch.stat().st_size,
                              patch_sha256=sha(patch)), indent=2))


if __name__ == '__main__':
    main()
