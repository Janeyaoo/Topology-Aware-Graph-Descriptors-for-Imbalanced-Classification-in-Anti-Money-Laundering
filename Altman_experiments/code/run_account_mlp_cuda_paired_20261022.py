"""Frozen-budget account MLP paired feature experiment.

CUDA changes execution only. Model, fit-only preprocessing, ordered search,
early stopping, seeds, CPU inference and threshold selection follow the original
2026-10-13 MLP. A fresh CUDA Base16 is included; old MPS scores are not pooled.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, fcntl, gc, hashlib, importlib.util, json, platform, resource
import shutil, signal, sys, tempfile, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Altman_experiments/results/account_mlp_cuda_paired_20261022'
PACKAGE = ROOT / 'Altman_experiments/results/account_graphsage_cuda_features_20261018/package'
DATASETS = ('hi_small', 'li_small')
ARMS = ('base16', 'base_g32', 'gmke44', 'gmkep56')
WIDTHS = dict(zip(ARMS, (16, 32, 44, 56)))
STOP_REQUESTED = False
MODEL = 'mlp'

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''): h.update(part)
    return h.hexdigest()
def ah(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def rel(path): return str(Path(path).resolve())
def sync_dir(directory):
    fd = os.open(str(directory), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
def atomic_bytes(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path); sync_dir(path.parent)
def jsave(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path); sync_dir(path.parent)
def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path); sync_dir(path.parent)
def atomic_parquet(path, frame):
    path = Path(path); temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    frame.to_parquet(temporary, index=False, compression='zstd')
    with temporary.open('rb') as stream: os.fsync(stream.fileno())
    temporary.replace(path); sync_dir(path.parent)
def load_torch(path, **kwargs):
    return torch.load(path, weights_only=False, **kwargs)
def verify_artifacts(items):
    for path, digest in items.items():
        assert sha(path) == digest, f'Artifact changed or incomplete: {path}'
def memory():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {'peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss * 1024)}
    if torch.cuda.is_available():
        out.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                   cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    return out
def configure():
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: assert torch.get_num_interop_threads() == 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, 'set_float32_matmul_precision'): torch.set_float32_matmul_precision('highest')
def stop_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
class SafeStop(Exception): pass
def worker_identity(ds, arm, seed, fold):
    start_ticks = boot_id = None
    if sys.platform.startswith('linux'):
        start_ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
        boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return {'pid': os.getpid(), 'start_ticks': start_ticks, 'boot_id': boot_id,
            'task': {'dataset': ds, 'feature_set': arm, 'seed': seed, 'fold': fold}}

class AccountMLP(nn.Module):
    def __init__(self, input_dimension, params):
        super().__init__(); self.name = MODEL; self.dropout = float(params['dropout'])
        hidden = int(params['hidden_dim'])
        self.layers = nn.ModuleList([nn.Linear(input_dimension, hidden), nn.Linear(hidden, 1)])
    def forward_mlp(self, features):
        hidden = torch.relu(self.layers[0](features))
        hidden = nn.functional.dropout(hidden, p=self.dropout, training=self.training)
        return self.layers[1](hidden).squeeze(-1)

def predict(model, X, rows, chunk=16384):
    """Preserve the original CPU float32/chunk16384 prediction path."""
    assert torch.get_num_threads() == 2 and chunk == 16384
    cpu = copy.deepcopy(model).to('cpu').eval()
    result = np.empty(len(rows), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            index = rows[start:start + chunk]
            logits = cpu.forward_mlp(torch.from_numpy(np.ascontiguousarray(X[index])))
            result[start:start + len(index)] = torch.sigmoid(logits).double().numpy()
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    return result
def state_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def cpu_state(model): return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
def rng_state(device):
    return {'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None}
def restore_rng(value, device):
    torch.set_rng_state(value['torch_cpu'])
    if str(device).startswith('cuda'): torch.cuda.set_rng_state_all(value['torch_cuda'])

def checkpoint_save(pointer, value):
    """Two content-addressed generations; pointer commits last, bounded retention."""
    pointer = Path(pointer); previous = read(pointer) if pointer.exists() else None
    temporary = pointer.with_name(pointer.name + f'.{os.getpid()}.writing')
    atomic_torch(temporary, value); digest = sha(temporary)
    artifact = pointer.with_name(pointer.name + f'.{digest}.pt')
    temporary.replace(artifact); sync_dir(pointer.parent)
    old = None if previous is None else previous['current']
    jsave(pointer, {'format_version': 1, 'current': {'path': rel(artifact), 'sha256': digest},
                   'previous': old, 'updated_at': now(), 'fit_binding_sha256': value['fit_binding_sha256']})
    keep = {artifact.name}
    if old: keep.add(Path(old['path']).name)
    for orphan in pointer.parent.glob(pointer.name + '.*.pt'):
        if orphan.name not in keep: orphan.unlink()
def checkpoint_load(pointer, binding):
    pointer = Path(pointer)
    if not pointer.exists(): return None
    index = read(pointer); assert index['fit_binding_sha256'] == binding
    current = index['current']; assert sha(current['path']) == current['sha256'], 'Checkpoint digest mismatch'
    value = load_torch(current['path'], map_location='cpu')
    assert value['fit_binding_sha256'] == binding
    return value
def fit(raw, y, fit_rows, early_rows, params, seed, device, path,
        protocol_sha256, input_binding_sha256, epochs=None, probe_epoch_limit=None,
        probe_interrupt_after_batches=None):
    """Preloaded CUDA inputs with exact recipe and resumable epoch/minibatch state."""
    path = Path(path); start = time.monotonic()
    binding = {'protocol_sha256': protocol_sha256, 'input_binding_sha256': input_binding_sha256,
        'raw_sha256': ah(raw), 'labels_sha256': ah(y), 'fit_rows_sha256': ah(fit_rows),
        'early_rows_sha256': None if early_rows is None else ah(early_rows), 'params': params,
        'seed': seed, 'input_dimension': raw.shape[1], 'device': device, 'epochs': epochs,
        'probe_epoch_limit': probe_epoch_limit}
    binding_hash = object_hash(binding)
    done = path.with_suffix('.fit_done.json'); pointer = path.with_suffix('.checkpoint.json')
    scaler = StandardScaler().fit(raw[fit_rows])
    X = np.ascontiguousarray(scaler.transform(raw), dtype=np.float32)
    assert np.isfinite(X).all()
    torch.manual_seed(seed)
    model = AccountMLP(raw.shape[1], params).to(device)
    if done.exists():
        receipt = read(done); assert receipt['fit_binding_sha256'] == binding_hash
        verify_artifacts(receipt['artifact_sha256'])
        value = load_torch(path, map_location='cpu'); model.load_state_dict(value['state_dict'])
        assert state_digest(model) == value['metadata']['model_digest']
        return model, X, value['metadata']
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['learning_rate']), weight_decay=1e-4)
    n = len(fit_rows); npos = int(y[fit_rows].sum()); assert 0 < npos < n
    weights = np.array([n / (2 * (n - npos)), n / (2 * npos)], dtype=np.float32)
    generator = np.random.default_rng(seed + 1000000)
    limit = int(epochs or 200)
    if probe_epoch_limit is not None: limit = int(probe_epoch_limit)
    assert probe_epoch_limit is not None or limit <= 200
    state = {'epoch': 1, 'cursor': 0, 'ordered': None, 'total_loss': 0., 'history': [],
        'best_auc': None, 'best_epoch': 0, 'best_state': None, 'finished': False,
        'elapsed_seconds': 0., 'batches_completed': 0}
    restored = checkpoint_load(pointer, binding_hash)
    if restored is not None:
        assert restored['fit_binding'] == binding
        model.load_state_dict(restored['model_state']); optimizer.load_state_dict(restored['optimizer_state'])
        # Adam.load_state_dict casts moments to parameter devices while retaining
        # its non-capturable step counter on CPU, just as a fresh optimizer does.
        state = restored['progress']; generator.bit_generator.state = restored['numpy_generator_state']
        restore_rng(restored['rng'], device)
    prior_seconds = state['elapsed_seconds']
    # Resident tensors eliminate one CPU-to-GPU feature/label copy per batch.
    resident_x = torch.from_numpy(X).to(device)
    resident_y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).to(device)
    resident_w = torch.from_numpy(np.ascontiguousarray(weights[y])).to(device)
    batches_this_call = 0
    def save():
        state['elapsed_seconds'] = prior_seconds + time.monotonic() - start
        checkpoint_save(pointer, {'fit_binding': binding, 'fit_binding_sha256': binding_hash,
            'model_state': cpu_state(model), 'optimizer_state': optimizer.state_dict(),
            'progress': state, 'numpy_generator_state': generator.bit_generator.state,
            'rng': rng_state(device)})
    while not state['finished'] and state['epoch'] <= limit:
        epoch = state['epoch']; model.train()
        if state['ordered'] is None:
            state['ordered'] = generator.permutation(fit_rows); state['cursor'] = 0; state['total_loss'] = 0.
        pending_losses = []
        def flush_losses():
            if pending_losses:
                # One synchronization for an epoch (or safe interruption), then
                # the original left-to-right Python float accumulation order.
                values = torch.stack([value for value, count in pending_losses]).cpu().numpy()
                for value, (_, count) in zip(values, pending_losses):
                    state['total_loss'] += float(value) * count
                pending_losses.clear()
        for offset in range(state['cursor'], n, 2048):
            rows = state['ordered'][offset:offset + 2048]
            index = torch.as_tensor(rows, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_mlp(resident_x[index])
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, resident_y[index], reduction='none') * resident_w[index]).mean()
            loss.backward(); optimizer.step()
            pending_losses.append((loss.detach(), len(rows)))
            state['cursor'] = offset + len(rows); state['batches_completed'] += 1; batches_this_call += 1
            if STOP_REQUESTED or (probe_interrupt_after_batches is not None and batches_this_call >= probe_interrupt_after_batches):
                flush_losses(); save(); raise SafeStop('Checkpoint persisted at a completed minibatch')
        flush_losses()
        record = {'epoch': epoch, 'weighted_seed_loss': state['total_loss'] / n,
                  'seconds': prior_seconds + time.monotonic() - start}
        if early_rows is not None:
            probability = predict(model, X, early_rows)
            auc = float(roc_auc_score(y[early_rows], probability)); record['early_auroc'] = auc
            if state['best_auc'] is None or auc > state['best_auc']:
                state['best_auc'] = auc; state['best_epoch'] = epoch; state['best_state'] = cpu_state(model)
            if epoch - state['best_epoch'] >= 30: state['finished'] = True
        else: state['best_epoch'] = epoch
        state['history'].append(record); state['ordered'] = None; state['cursor'] = 0
        state['epoch'] += 1
        if epoch == limit: state['finished'] = True
        save()
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({'event': 'fit_epoch', 'path': rel(path), **record}), flush=True)
        if STOP_REQUESTED: raise SafeStop('Checkpoint persisted at an epoch boundary')
    if early_rows is not None: model.load_state_dict(state['best_state'])
    metadata = {'model': MODEL, 'params': params, 'device': device, 'random_state': seed,
        'scaler_fit_rows_sha256': ah(fit_rows), 'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(), 'scaler_var': scaler.var_.tolist(),
        'scaler_n_samples_seen': int(scaler.n_samples_seen_), 'fit_rows_sha256': ah(fit_rows),
        'early_rows_sha256': None if early_rows is None else ah(early_rows),
        'fit_n': n, 'fit_positive': npos, 'class_weights': weights.tolist(), 'seed_only_loss': True,
        'selected_epochs': state['best_epoch'], 'epochs_run': len(state['history']),
        'best_early_auroc': state['best_auc'], 'history': state['history'],
        'model_digest': state_digest(model), 'seconds': state['elapsed_seconds'], **memory(),
        'input_dimension': raw.shape[1], 'prediction_threads': 2, 'prediction_chunk_size': 16384,
        'protocol_sha256': protocol_sha256, 'input_binding_sha256': input_binding_sha256,
        'fit_binding': binding, 'fit_binding_sha256': binding_hash, 'checkpoint_resumed': restored is not None,
        'preloaded_tensors': True, 'autocast': False, 'tf32': False}
    atomic_torch(path, {'state_dict': cpu_state(model), 'metadata': metadata})
    jsave(path.with_suffix('.json'), metadata)
    jsave(done, {'completed_at': now(), 'fit_binding_sha256': binding_hash,
        'artifact_sha256': {rel(item): sha(item) for item in (path, path.with_suffix('.json'))}})
    return model, X, metadata

def best_f1_threshold(y, probability):
    order = np.argsort(-probability, kind='mergesort'); scores = probability[order]
    positives = np.cumsum(y[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    thresholds = scores[ends][::-1]
    values = (2 * positives[ends])[::-1] / (ends + 1 + int(y.sum()))[::-1]
    position = int(np.argmax(values)); return float(thresholds[position]), float(values[position])
def metrics(y, probability, threshold):
    predicted = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return predicted, dict(precision=float(precision_score(y, predicted, zero_division=0)),
        recall=float(recall_score(y, predicted, zero_division=0)), f1=float(f1_score(y, predicted, zero_division=0)),
        auc=float(roc_auc_score(y, probability)), ap=float(average_precision_score(y, probability)),
        accuracy=float(accuracy_score(y, predicted)), balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        specificity=float(tn / (tn + fp)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        n_test=len(y), n_positive=int(y.sum()))
def arm_columns(names, arm):
    return [names['columns'][value] if isinstance(value, int) else value for value in names['feature_sets'][arm]]
def validate_splits(arrays, n):
    for key, value in arrays.items():
        assert value.ndim == 1 and np.issubdtype(value.dtype, np.integer), key
        assert len(value) == len(np.unique(value)) and (value >= 0).all() and (value < n).all(), key
    def partition(parent, left, right):
        np.testing.assert_array_equal(np.sort(np.r_[left, right]), np.sort(parent))
    partition(np.arange(n), arrays['outer_train'], arrays['outer_test'])
    partition(arrays['outer_train'], arrays['development'], arrays['threshold_validation'])
    for j in range(3):
        partition(arrays['development'], arrays[f'inner_train_{j}'], arrays[f'inner_score_{j}'])
        partition(arrays[f'inner_train_{j}'], arrays[f'early_fit_{j}'], arrays[f'early_validation_{j}'])
def expected_candidates():
    return [{'candidate_id': i, 'params': {'hidden_dim': h, 'learning_rate': lr, 'dropout': drop}}
        for i, (h, lr, drop) in enumerate((h, lr, drop) for h in (64, 128, 256) for lr in (.001, .003) for drop in (0., .2))]
def expected_input_map(path):
    value = read(path)
    if 'checks' in value:
        return {key: item['expected_sha256'] for key, item in value['checks'].items()}
    return value
def prepare(expected_binding):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'protocol.json').exists():
            p = registered(); assert Path(p['input_package']).resolve() == PACKAGE.resolve()
            print(json.dumps({'event': 'already_registered', 'protocol_sha256': sha(OUT / 'protocol.json')})); return
        known = expected_input_map(expected_binding)
        candidates = read(PACKAGE / 'candidates.json')
        assert candidates['candidates'] == expected_candidates()
        original = ROOT / 'Altman_experiments/code/run_account_eswa_supplement_neural_20261013.py'
        common = ROOT / 'Altman_experiments/code/run_account_feature_mechanism.py'
        design = ROOT / 'Altman_experiments/results/eswa_supplement_design_20261013'
        old = ROOT / 'Altman_experiments/results/account_gmkep56_tuned_20261008'
        sources = [original, common, design / 'neural_candidates.json', design / 'protocol_design.json',
                   old / 'protocol.json', old / 'split_manifest.json', Path(expected_binding).resolve()]
        for source in sources: assert source.is_file(), source
        assert read(design / 'neural_candidates.json')['candidates'] == candidates['candidates']
        binding = {rel(source): sha(source) for source in sources}; datasets = {}
        snapshots = OUT / 'source_snapshots'; snapshots.mkdir(exist_ok=True)
        for source in (original, common): shutil.copy2(source, snapshots / source.name)
        for ds in DATASETS:
            folder = PACKAGE / 'inputs' / ds; names = read(folder / 'names.json')
            features = np.load(folder / 'features.npy', mmap_mode='r'); y = np.load(folder / 'labels.npy')
            identity = pd.read_parquet(folder / 'accounts.parquet')
            assert len(names['columns']) == len(set(names['columns'])) == 56
            assert features.dtype == np.float64 and features.shape == (len(y), 56) and np.isfinite(features).all()
            assert set(np.unique(y)) == {0, 1} and y.dtype == np.int8
            np.testing.assert_array_equal(identity.row_index.to_numpy(), np.arange(len(y)))
            assert identity.account_id.is_unique
            base = names['base16']; assert len(base) == 16
            for arm in ARMS:
                columns = arm_columns(names, arm)
                assert len(columns) == len(set(columns)) == WIDTHS[arm] and set(base).issubset(columns)
                if arm == 'base16': assert columns == base
                for sub in ('search', 'folds', 'models', 'calibration_models', 'calibration', 'predictions'):
                    (OUT / 'runs' / ds / arm / sub).mkdir(parents=True, exist_ok=True)
            assert (features[:, [names['columns'].index(col) for col in base]] >= 0).all()
            files = [folder / item for item in ('features.npy', 'labels.npy', 'names.json', 'accounts.parquet')]
            for seed in range(5):
                tests = []
                for fold in range(5):
                    file = folder / 'splits' / f'seed{seed}_fold{fold}.npz'
                    with np.load(file) as z: arrays = {key: z[key] for key in z.files}
                    validate_splits(arrays, len(y)); tests.append(arrays['outer_test']); files.append(file)
                np.testing.assert_array_equal(np.sort(np.concatenate(tests)), np.arange(len(y)))
            for file in [PACKAGE / 'candidates.json', *files]:
                relative = str(file.relative_to(PACKAGE))
                assert relative in known and sha(file) == known[relative], f'Frozen input mismatch: {relative}'
                binding[rel(file)] = known[relative]
            manifest = {'dataset': ds, 'input_files': {rel(file): sha(file) for file in files},
                'n_samples': len(y), 'n_positive': int(y.sum()), 'base16': base,
                'feature_sets': {arm: arm_columns(names, arm) for arm in ARMS},
                'labels_array_sha256': ah(y),
                'account_ids_hash_sha256': ah(pd.util.hash_pandas_object(identity.account_id, index=False).to_numpy())}
            jsave(OUT / f'input_manifest_{ds}.json', manifest)
            datasets[ds] = {**manifest, 'input_binding_sha256': sha(OUT / f'input_manifest_{ds}.json')}
            del features, identity; gc.collect()
        jsave(OUT / 'candidates.json', candidates)
        binding[rel(OUT / 'candidates.json')] = sha(OUT / 'candidates.json')
        binding.update({rel(file): sha(file) for file in snapshots.iterdir() if file.is_file()})
        jsave(OUT / 'source_binding.json', binding)
        protocol = {'registered_at': now(), 'runner_sha256': sha(__file__), 'device': 'cuda', 'model': MODEL,
            'input_package': rel(PACKAGE), 'datasets': datasets, 'feature_sets': list(ARMS),
            'source_binding_sha256': sha(OUT / 'source_binding.json'),
            'preprocessing': 'BASE16 float64 log1p then float32; other columns raw float32; fit-only StandardScaler on all selected columns',
            'architecture': 'Linear(input_dimension,hidden)-ReLU-dropout-Linear(hidden,1); hidden64/128/256',
            'selection': '12 ordered original candidates x3 frozen splits; early unweighted AUROC strict >, patience30/max200; mean3 unweighted AP, lowest candidate ID tie; median3 best epochs',
            'threshold': 'development fit; threshold_validation max F1 with lowest threshold on tie; same-seed/epochs outer_train refit; fixed0.5 also saved',
            'weights': 'fit-only n/(2*nclass), weighted BCE mean; Adam weight_decay1e-4; batch2048',
            'execution_settings': {'cpu_threads': 2, 'interop_threads': 1, 'prediction_chunk_size': 16384,
                'fit_dtype': 'float32', 'prediction_device': 'cpu', 'prediction_dtype': 'float32->float64',
                'autocast': False, 'tf32': False, 'deterministic_algorithms': True,
                'preloaded_tensors': True, 'checkpoint': 'every epoch and SIGTERM safe minibatch; two digest-checked generations'},
            'budget': {'arms_per_dataset': 4, 'outer_folds_per_setting': 25,
                'outer_jobs': 200, 'inner_candidate_scores_per_outer': 36, 'stages_per_outer': 4},
            'seeds': {'search': '40000+1000*seed+10*fold+inner', 'stage': '100*seed+fold', 'permutation': 'fit_seed+1000000'},
            'historical_mps_scores_reused': False, 'graph_read_by_mlp': False,
            'environment': {'torch': torch.__version__, 'numpy': np.__version__,
                'platform': platform.platform(), 'cuda_runtime': torch.version.cuda}}
        shutil.copy2(__file__, OUT / 'runner_snapshot.py')
        jsave(OUT / 'protocol.json', protocol)
        print(json.dumps({'event': 'prepared', 'protocol_sha256': sha(OUT / 'protocol.json')}), flush=True)
def registered(ds=None):
    protocol = read(OUT / 'protocol.json')
    assert protocol['runner_sha256'] == sha(__file__), 'Registered runner changed'
    assert protocol['source_binding_sha256'] == sha(OUT / 'source_binding.json')
    verify_artifacts(read(OUT / 'source_binding.json'))
    assert protocol['device'] == 'cuda' and protocol['model'] == MODEL
    assert protocol['environment']['torch'] == torch.__version__, 'Registered Torch version changed'
    assert protocol['environment']['numpy'] == np.__version__, 'Registered NumPy version changed'
    assert protocol['environment']['cuda_runtime'] == torch.version.cuda, 'Registered CUDA runtime changed'
    for dataset in DATASETS if ds is None else (ds,):
        file = OUT / f'input_manifest_{dataset}.json'
        assert sha(file) == protocol['datasets'][dataset]['input_binding_sha256']
    return protocol
def load_data(ds, arm, seed, fold, protocol):
    folder = Path(protocol['input_package']) / 'inputs' / ds
    names = read(folder / 'names.json'); columns = arm_columns(names, arm)
    original = np.load(folder / 'features.npy', mmap_mode='r')
    raw = np.ascontiguousarray(original[:, [names['columns'].index(col) for col in columns]], dtype=np.float64)
    for index, column in enumerate(columns):
        if column in names['base16']: raw[:, index] = np.log1p(raw[:, index])
    raw = np.ascontiguousarray(raw, dtype=np.float32); assert np.isfinite(raw).all()
    y = np.load(folder / 'labels.npy'); identity = pd.read_parquet(folder / 'accounts.parquet')
    with np.load(folder / 'splits' / f'seed{seed}_fold{fold}.npz') as z: arrays = {key: z[key] for key in z.files}
    validate_splits(arrays, len(y))
    return raw, y, identity, arrays
def model_artifacts(path):
    return [path, path.with_suffix('.json'), path.with_suffix('.fit_done.json')]
def check_record(record, protocol_hash, input_hash):
    assert record['protocol_sha256'] == protocol_hash and record['input_binding_sha256'] == input_hash
    verify_artifacts(record['artifact_sha256'])
def search(ds, arm, seed, fold, raw, y, arrays, protocol):
    protocol_hash = sha(OUT / 'protocol.json'); input_hash = protocol['datasets'][ds]['input_binding_sha256']
    directory = OUT / 'runs' / ds / arm / 'search' / f'mlp_seed{seed}_fold{fold}'
    directory.mkdir(parents=True, exist_ok=True); result_path = directory / 'search_result.json'
    if result_path.exists():
        result = read(result_path); check_record(result, protocol_hash, input_hash)
        assert (result['dataset'], result['feature_set'], result['seed'], result['fold']) == (ds, arm, seed, fold)
        return result
    candidates = read(OUT / 'candidates.json')['candidates']; records = []; record_files = []
    for candidate in candidates:
        for j in range(3):
            recpath = directory / f"candidate{candidate['candidate_id']}_inner{j}_record.json"
            if recpath.exists():
                record = read(recpath); check_record(record, protocol_hash, input_hash)
                assert (record['dataset'], record['feature_set'], record['seed'], record['fold'], record['candidate_id'], record['inner_fold']) == (ds, arm, seed, fold, candidate['candidate_id'], j)
                assert record['params'] == candidate['params']
            else:
                path = directory / f"candidate{candidate['candidate_id']}_inner{j}.pt"
                fit_rows, early, score_rows = (arrays[f'{key}_{j}'] for key in ('early_fit', 'early_validation', 'inner_score'))
                model, X, meta = fit(raw, y, fit_rows, early, candidate['params'],
                    40000 + 1000 * seed + 10 * fold + j, 'cuda', path, protocol_hash, input_hash)
                tick = time.monotonic(); score = predict(model, X, score_rows)
                scorepath = path.with_suffix('.npz'); atomic_npz(scorepath, row_index=score_rows, y_true=y[score_rows], y_score=score)
                record = {'dataset': ds, 'model': MODEL, 'feature_set': arm, 'seed': seed, 'fold': fold,
                    'candidate_id': candidate['candidate_id'], 'inner_fold': j, 'params': candidate['params'],
                    'ap': float(average_precision_score(y[score_rows], score)), 'auc': float(roc_auc_score(y[score_rows], score)),
                    'selected_epochs': meta['selected_epochs'], 'fit_seconds': meta['seconds'], 'predict_seconds': time.monotonic() - tick,
                    'fit_rows_sha256': ah(fit_rows), 'early_rows_sha256': ah(early), 'score_rows_sha256': ah(score_rows),
                    'raw_score_sha256': ah(score), 'model_path': rel(path), 'score_path': rel(scorepath),
                    'model_digest': meta['model_digest'], 'input_dimension': raw.shape[1], 'fit_metadata': meta,
                    'protocol_sha256': protocol_hash, 'input_binding_sha256': input_hash,
                    'artifact_sha256': {rel(file): sha(file) for file in [*model_artifacts(path), scorepath]}}
                jsave(recpath, record)
                print(json.dumps({'event': 'candidate_complete', **{key: record[key] for key in ('dataset', 'feature_set', 'seed', 'fold', 'candidate_id', 'inner_fold', 'ap', 'selected_epochs')}}), flush=True)
                del model, X, score; gc.collect()
            records.append(record); record_files.append(recpath)
    summaries = []
    for candidate in candidates:
        rows = [record for record in records if record['candidate_id'] == candidate['candidate_id']]
        assert len(rows) == 3
        summaries.append({'candidate_id': candidate['candidate_id'], 'mean_ap': float(np.mean([row['ap'] for row in rows])),
            'std_ap': float(np.std([row['ap'] for row in rows], ddof=1)), 'epochs': int(np.median([row['selected_epochs'] for row in rows]))})
    selected = min(summaries, key=lambda row: (-row['mean_ap'], row['candidate_id']))
    result = {'dataset': ds, 'model': MODEL, 'feature_set': arm, 'seed': seed, 'fold': fold,
        'candidate_records': records, 'candidate_summary': summaries, 'selected_candidate_id': selected['candidate_id'],
        'selected_mean_ap': selected['mean_ap'], 'selected_epochs': selected['epochs'], 'anchor_epochs': summaries[0]['epochs'],
        'protocol_sha256': protocol_hash, 'input_binding_sha256': input_hash,
        'artifact_sha256': {**{path: digest for record in records for path, digest in record['artifact_sha256'].items()},
                            **{rel(file): sha(file) for file in record_files}}}
    jsave(result_path, result); return result
def stages(ds, arm, seed, fold, strategy, cid, epochs, params, raw, y, arrays, identity, protocol):
    folder = OUT / 'runs' / ds / arm; stem = f'mlp_seed{seed}_fold{fold}_{strategy}'
    protocol_hash = sha(OUT / 'protocol.json'); input_hash = protocol['datasets'][ds]['input_binding_sha256']
    D, T, train, test = (arrays[key] for key in ('development', 'threshold_validation', 'outer_train', 'outer_test'))
    modelpath = folder / 'calibration_models' / f'{stem}.pt'
    model, X, calmeta = fit(raw, y, D, None, params, 100 * seed + fold, 'cuda', modelpath, protocol_hash, input_hash, epochs)
    cal_score = predict(model, X, T); threshold, calf1 = best_f1_threshold(y[T], cal_score)
    calpath = folder / 'calibration' / f'{stem}.parquet'
    frame = identity.iloc[T][['row_index', 'account_id']].copy(); frame['y_true'] = y[T]; frame['y_score'] = cal_score
    atomic_parquet(calpath, frame); del model, X, frame; gc.collect()
    finalpath = folder / 'models' / f'{stem}.pt'
    model, X, finalmeta = fit(raw, y, train, None, params, 100 * seed + fold, 'cuda', finalpath, protocol_hash, input_hash, epochs)
    score = predict(model, X, test); predpath = folder / 'predictions' / f'{stem}.parquet'
    frame = identity.iloc[test][['row_index', 'account_id']].copy(); frame['y_true'] = y[test]; frame['y_score'] = score
    records = []
    for policy, cutoff in (('fixed_0.5', .5), ('inner_max_f1', threshold)):
        predicted, values = metrics(y[test], score, cutoff)
        frame['y_pred_fixed' if policy == 'fixed_0.5' else 'y_pred_inner'] = predicted
        records.append({'dataset': ds, 'model': MODEL, 'feature_set': arm, 'seed': seed, 'fold': fold,
            'strategy': strategy, 'threshold_policy': policy, 'threshold': cutoff,
            'selected_candidate_id': cid, 'selected_epochs': epochs, **values})
    atomic_parquet(predpath, frame)
    files = [*model_artifacts(modelpath), calpath, *model_artifacts(finalpath), predpath]
    result = {'strategy': strategy, 'candidate_id': cid, 'selected_epochs': epochs, 'params': params, 'threshold': threshold,
        'threshold_validation_f1': calf1, 'threshold_validation_ap': float(average_precision_score(y[T], cal_score)),
        'model_random_state': 100 * seed + fold, 'fit_rows_sha256': ah(train), 'calibration_fit_rows_sha256': ah(D),
        'threshold_rows_sha256': ah(T), 'outer_test_rows_sha256': ah(test),
        'calibration_model_path': rel(modelpath), 'model_path': rel(finalpath), 'calibration_path': rel(calpath),
        'prediction_path': rel(predpath), 'calibration_metadata': calmeta, 'final_metadata': finalmeta,
        'raw_score_sha256': ah(score), 'metrics': records, 'protocol_sha256': protocol_hash,
        'input_binding_sha256': input_hash, 'artifact_sha256': {rel(file): sha(file) for file in files}}
    del model, X, score, cal_score, frame; gc.collect()
    return result
def run_fold(ds, arm, seed, fold):
    assert torch.cuda.is_available(), 'Formal fitting requires CUDA; no fallback'
    protocol = registered(ds); probe = read(OUT / 'probe_cuda.json')
    assert probe['passed'] and probe['runner_sha256'] == sha(__file__) and probe['device'] == 'cuda'
    assert probe['protocol_sha256'] == sha(OUT / 'protocol.json'), 'CUDA probe belongs to another protocol'
    folder = OUT / 'runs' / ds / arm; stem = f'mlp_seed{seed}_fold{fold}'
    with (folder / 'folds' / f'{stem}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = folder / 'folds' / f'{stem}.json'; done = folder / 'folds' / f'{stem}_DONE.json'
        job = folder / 'folds' / f'{stem}_job.json'
        protocol_hash = sha(OUT / 'protocol.json'); input_hash = protocol['datasets'][ds]['input_binding_sha256']
        if done.exists():
            receipt = read(done); check_record(receipt, protocol_hash, input_hash)
            assert (receipt['dataset'], receipt['feature_set'], receipt['seed'], receipt['fold']) == (ds, arm, seed, fold)
            assert receipt['result_sha256'] == sha(path)
            print(json.dumps({'event': 'already_complete', 'path': rel(path)})); return
        manifest = {'dataset': ds, 'feature_set': arm, 'model': MODEL, 'seed': seed, 'fold': fold,
            'status': 'running', 'pid': os.getpid(), 'started_at': now(), 'protocol_sha256': protocol_hash,
            'worker_identity': worker_identity(ds, arm, seed, fold),
            'source_binding_sha256': protocol['source_binding_sha256'], 'input_binding_sha256': input_hash}
        jsave(job, manifest)
        try:
            raw, y, identity, arrays = load_data(ds, arm, seed, fold, protocol)
            result = search(ds, arm, seed, fold, raw, y, arrays, protocol)
            candidates = read(OUT / 'candidates.json')['candidates']; results = []
            for strategy, cid, epochs in (('same_protocol_anchor', 0, result['anchor_epochs']),
                    ('tuned', result['selected_candidate_id'], result['selected_epochs'])):
                partial = folder / 'folds' / f'{stem}_{strategy}_partial.json'
                if partial.exists():
                    value = read(partial); check_record(value, protocol_hash, input_hash)
                    assert value['strategy'] == strategy and value['candidate_id'] == cid
                    assert value['selected_epochs'] == epochs and value['params'] == candidates[cid]['params']
                else:
                    value = stages(ds, arm, seed, fold, strategy, cid, epochs, candidates[cid]['params'], raw, y, arrays, identity, protocol)
                    jsave(partial, value)
                results.append(value)
            artifacts = {**result['artifact_sha256'], **{key: digest for item in results for key, digest in item['artifact_sha256'].items()}}
            artifacts[rel(folder / 'search' / stem / 'search_result.json')] = sha(folder / 'search' / stem / 'search_result.json')
            for strategy in ('same_protocol_anchor', 'tuned'):
                file = folder / 'folds' / f'{stem}_{strategy}_partial.json'; artifacts[rel(file)] = sha(file)
            saved = {'dataset': ds, 'model': MODEL, 'feature_set': arm, 'input_dimension': raw.shape[1],
                'seed': seed, 'fold': fold, 'search': result, 'strategies': results,
                'logical_candidate_evaluations': 36, 'logical_strategy_models': 4, 'artifact_sha256': artifacts,
                'protocol_sha256': protocol_hash, 'input_binding_sha256': input_hash,
                'source_binding_sha256': protocol['source_binding_sha256'], 'completed_at': now()}
            jsave(path, saved); check_record(saved, protocol_hash, input_hash)
            manifest.update(status='complete', pid=None, exit_code=0, finished_at=now(), result_path=rel(path), result_sha256=sha(path), **memory())
            jsave(job, manifest)
            jsave(done, {**manifest, 'passed': True, 'artifact_sha256': {**artifacts, rel(path): sha(path)}})
            print(json.dumps({'event': 'outer_complete', **{key: saved[key] for key in ('dataset', 'feature_set', 'seed', 'fold')}}), flush=True)
        except BaseException as error:
            manifest.update(status='checkpointed' if isinstance(error, SafeStop) else 'failed', pid=None,
                exit_code=75 if isinstance(error, SafeStop) else 1,
                error=repr(error), traceback=traceback.format_exc(), updated_at=now(), **memory())
            jsave(job, manifest); raise

def probe(device):
    """Synthetic numeric and restart checks; never creates experiment scores."""
    if device == 'cuda':
        assert torch.cuda.is_available(); registered()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123); reports = []
    reference_path = ROOT / 'Altman_experiments/code/run_account_eswa_supplement_neural_20261013.py'
    spec = importlib.util.spec_from_file_location('frozen_mlp_reference', reference_path)
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    with tempfile.TemporaryDirectory(prefix='mlp_probe_', dir=OUT) as name:
        directory = Path(name)
        y = np.tile(np.array([0, 0, 0, 1], dtype=np.int8), 1300)
        fit_rows = np.arange(4500); early = np.arange(4500, 4800); score_rows = np.arange(4800, 5200)
        for dimension in (16, 32, 44, 56):
            raw = rng.normal(size=(len(y), dimension)).astype(np.float32)
            params = {'hidden_dim': 64, 'learning_rate': .001, 'dropout': .2}
            torch.manual_seed(84); candidate = AccountMLP(dimension, params)
            if dimension == 16:
                torch.manual_seed(84); old = reference.AccountNetwork('mlp', params)
                assert state_digest(candidate) == reference.state_digest(old)
                X = np.ascontiguousarray(StandardScaler().fit(raw[fit_rows]).transform(raw), dtype=np.float32)
                np.testing.assert_array_equal(predict(candidate, X, score_rows), reference.predict(old, X, [], score_rows))
                # Complete two-epoch CPU trajectory, including dropout, Adam and seed-loss weights.
                reference.rel = lambda value: str(value)
                reference.jsave = jsave
                old_model, old_X, old_meta = reference.fit('mlp', raw, y, [], fit_rows, None, params, 84, 'cpu', directory / 'reference.pt', epochs=2)
                new_model, new_X, new_meta = fit(raw, y, fit_rows, None, params, 84, 'cpu', directory / 'reference_new.pt', 'probe', 'probe', epochs=2)
                np.testing.assert_array_equal(old_X, new_X)
                assert state_digest(old_model) == state_digest(new_model)
                assert old_meta['class_weights'] == new_meta['class_weights']
                assert old_meta['scaler_mean'] == new_meta['scaler_mean']
                np.testing.assert_array_equal([row['weighted_seed_loss'] for row in old_meta['history']], [row['weighted_seed_loss'] for row in new_meta['history']])
            full, full_X, full_meta = fit(raw, y, fit_rows, early, params, 52, device, directory / f'full{dimension}.pt', 'probe', 'probe', probe_epoch_limit=3)
            interrupted = directory / f'resumed{dimension}.pt'
            try:
                fit(raw, y, fit_rows, early, params, 52, device, interrupted, 'probe', 'probe', probe_epoch_limit=3, probe_interrupt_after_batches=2)
                raise AssertionError('Interruption not exercised')
            except SafeStop: pass
            resumed, resumed_X, resumed_meta = fit(raw, y, fit_rows, early, params, 52, device, interrupted, 'probe', 'probe', probe_epoch_limit=3)
            assert state_digest(full) == state_digest(resumed)
            np.testing.assert_array_equal(predict(full, full_X, score_rows), predict(resumed, resumed_X, score_rows))
            for key in ('weighted_seed_loss', 'early_auroc'):
                np.testing.assert_array_equal([row[key] for row in full_meta['history']], [row[key] for row in resumed_meta['history']])
            assert full_meta['selected_epochs'] == resumed_meta['selected_epochs']
            pointer = interrupted.with_suffix('.checkpoint.json'); checkpoint = read(pointer)
            assert len(list(pointer.parent.glob(pointer.name + '.*.pt'))) <= 2
            assert sha(checkpoint['current']['path']) == checkpoint['current']['sha256']
            reports.append({'input_dimension': dimension, 'interrupted_minibatch_resume_bitwise_equal': True,
                'scores_bitwise_equal': True, 'history_bitwise_equal_excluding_time': True})
    report = {'passed': True, 'device': device, 'runner_sha256': sha(__file__), 'reference_sha256': sha(reference_path),
        'protocol_sha256': sha(OUT / 'protocol.json') if (OUT / 'protocol.json').exists() else None,
        'synthetic_only': True, 'outer_test_evaluated': False, 'reports': reports,
        'cpu16_reference_model_scaler_seed_loss_prediction_bitwise_equal': True,
        'execution_settings': {'cpu_threads': torch.get_num_threads(), 'tf32': False, 'autocast': False},
        'environment': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda}, 'finished_at': now()}
    jsave(OUT / f'probe_{device}.json', report)
    print(json.dumps(report), flush=True)

def main():
    global ROOT, OUT, PACKAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'probe', 'run'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--input-package', type=Path, default=PACKAGE)
    parser.add_argument('--expected-input-binding', type=Path)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--feature-set', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=range(5))
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(); ROOT = args.root.resolve(); OUT = args.out.resolve(); PACKAGE = args.input_package.resolve()
    configure(); signal.signal(signal.SIGTERM, stop_handler); signal.signal(signal.SIGINT, stop_handler)
    if args.command == 'prepare':
        if args.expected_input_binding is None: parser.error('prepare requires --expected-input-binding from an accepted frozen package receipt')
        prepare(args.expected_input_binding)
    elif args.command == 'probe': probe(args.device)
    else:
        if args.device != 'cuda': parser.error('formal run requires --device cuda')
        if any(value is None for value in (args.dataset, args.feature_set, args.seed, args.fold)):
            parser.error('run requires dataset, feature-set, seed and fold')
        run_fold(args.dataset, args.feature_set, args.seed, args.fold)
if __name__ == '__main__':
    try: main()
    except SafeStop as error:
        print(json.dumps({'event': 'checkpointed_exit', 'error': str(error)}), flush=True); sys.exit(75)
