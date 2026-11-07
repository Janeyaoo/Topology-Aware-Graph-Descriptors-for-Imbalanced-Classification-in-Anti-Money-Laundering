"""Explicit checkpoint execution adapter for the unchanged3830 scientific runner.

The original algorithm source and registered protocol remain immutable. Both
this adapter and the epoch engine must be accepted and bound by a separate
execution manifest. New metadata records the actual recovery execution sources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

BASE_SHA = '3830ebf59298f86e4cef4fa8c715e02fa37b8d75f35e6be0f7618e178f10930e'
SCRIPT = Path(__file__).resolve()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def inside(root, value):
    path = Path(value)
    path = (root / path if not path.is_absolute() else path).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Missing or escaping execution source: ' + str(value))
    return path


def validate_execution(root, manifest_path):
    root = Path(root).resolve()
    manifest_path = inside(root, manifest_path)
    manifest = read(manifest_path)
    require(manifest.get('schema_version') == 1 and manifest.get('status') == 'accepted',
            'Checkpoint execution requires an accepted schema1 manifest')
    require(manifest.get('root', str(root)) == str(root), 'Execution root differs')
    require(manifest['base_runner_sha256'] == BASE_SHA, 'Only the frozen3830 algorithm is supported')
    require(sha(root / 'protocol.json') == manifest['protocol_sha256'], 'Scientific protocol changed')
    protocol = read(root / 'protocol.json')
    require(protocol['runner_sha256'] == BASE_SHA and sha(root / 'runner_snapshot.py') == BASE_SHA,
            'Registered algorithm/snapshot differs')
    require(sha(root / 'source_binding.json') == protocol['source_binding_sha256'], 'Input source binding changed')
    if 'source_binding_sha256' in manifest:
        require(manifest['source_binding_sha256'] == protocol['source_binding_sha256'], 'Execution input binding differs')
    names = ('base_runner', 'engine', 'adapter', 'qa_report', 'baseline_inventory', 'provenance_auditor')
    files = {}
    for name in names:
        path = inside(root, manifest[name + '_path'])
        require(sha(path) == manifest[name + '_sha256'], 'Execution source changed: ' + name)
        if 'bindings' in manifest:
            binding = manifest['bindings'][name]
            require(inside(root, binding['path']) == path and binding['sha256'] == manifest[name + '_sha256'],
                    'Flat/nested execution binding differs: ' + name)
        files[name] = path
    require(files['adapter'] == SCRIPT and sha(SCRIPT) == manifest['adapter_sha256'], 'Actual adapter differs')
    require(sha(files['base_runner']) == BASE_SHA, 'Actual frozen algorithm differs')
    report = read(files['qa_report'])
    require(report.get('passed') is True, 'Checkpoint CUDA recovery acceptance has not passed')
    require(report.get('device') == 'cuda', 'Production checkpoint recovery requires target CUDA validation')
    require(report.get('engine_sha256') == manifest['engine_sha256'], 'QA engine source differs')
    require(report.get('base_runner_sha256') == BASE_SHA, 'QA algorithm source differs')
    if 'adapter_sha256' in report:
        require(report['adapter_sha256'] == manifest['adapter_sha256'], 'QA adapter source differs')
    probe = read(root / 'probe.json')
    require(probe.get('passed') is True and probe.get('device') == 'cuda' and probe.get('runner_sha256') == BASE_SHA,
            'Original CUDA algorithm admission is missing')
    return manifest_path, manifest, files


def import_file(name, path):
    require(name not in sys.modules, 'Execution module already loaded: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Cannot load accepted execution source')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execution-manifest', type=Path)
    args, scientific_args = parser.parse_known_args()
    require(scientific_args and scientific_args[0] == 'run-fold', 'Adapter only executes registered formal run-fold tasks')
    root = args.root.resolve()
    selected = args.execution_manifest or os.environ.get('FLOWTOPO_EXECUTION_MANIFEST')
    require(selected is not None, 'Explicit recovery execution manifest is required')
    manifest_path, manifest, files = validate_execution(root, selected)
    expected_cache = (root / 'scheduler/numba_cache_checkpoint').resolve()
    require(expected_cache.is_relative_to(root) and expected_cache == Path(os.environ.get('NUMBA_CACHE_DIR', '')).resolve(),
            'Formal checkpoint adapter must use its isolated registered Numba cache')
    base = import_file('run_account_graphsage_cuda_optimized_20261018', files['base_runner'])
    base.ROOT = root
    base.OUT = root / 'runs'
    engine = import_file('account_graphsage_epoch_checkpoint_20261018', files['engine'])
    base.fit = engine.make_checkpoint_fit(base, manifest_path)
    print(json.dumps(dict(event='checkpoint_execution_admitted', protocol_sha256=manifest['protocol_sha256'],
                          manifest_sha256=sha(manifest_path), base_runner_sha256=BASE_SHA,
                          engine_sha256=manifest['engine_sha256'], adapter_sha256=manifest['adapter_sha256'])), flush=True)
    sys.argv = [str(files['base_runner']), '--root', str(root), *scientific_args]
    base.main()


if __name__ == '__main__':
    main()
