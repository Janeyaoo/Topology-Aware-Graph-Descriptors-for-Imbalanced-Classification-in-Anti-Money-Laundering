"""Durable, exact epoch-boundary recovery for the frozen 3830 GraphSAGE fit.

This execution extension does not change the frozen scientific runner. The
adapter injects ``make_checkpoint_fit(base, manifest_path)`` as ``base.fit``.
Only complete epoch boundaries are recoverable; no partial minibatch is reused.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

BASE_SHA = '3830ebf59298f86e4cef4fa8c715e02fa37b8d75f35e6be0f7618e178f10930e'
SCHEMA = 1
EPOCH_COMMITTED_HOOK = None  # QA only: called after a fully durable generation.


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def durable_json(path, value):
    _atomic(path, lambda stream: stream.write(canonical(value) + b'\n'))


def durable_torch(path, value):
    _atomic(path, lambda stream: torch.save(value, stream))


def _torch_load(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=False)


def _within(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _read(path):
    return json.loads(Path(path).read_text())


def _rng_state(rng, is_cuda):
    return {'generator': copy.deepcopy(rng.bit_generator.state),
            'python': random.getstate(), 'numpy_global': np.random.get_state(),
            'torch_cpu': torch.get_rng_state().clone(),
            'torch_cuda_all': [value.clone() for value in torch.cuda.get_rng_state_all()] if is_cuda else None}


def _restore_rng(state, rng, is_cuda):
    # Deliberately last: model/Adam/scaler/device reconstruction precedes this.
    rng.bit_generator.state = state['generator']
    random.setstate(state['python'])
    np.random.set_state(state['numpy_global'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if is_cuda:
        assert len(state['torch_cuda_all']) == torch.cuda.device_count(), 'CUDA device count changed'
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda_all']])


def _load_generation(directory, context_sha, binding):
    """Use the newest legal generation; a corrupt latest falls back one epoch."""
    receipts = sorted(directory.glob('epoch*.receipt.json'), reverse=True)
    if not receipts:
        if (directory / 'fit_context.json').exists() or (directory / 'LATEST.json').exists() or list(directory.glob('epoch*.pt')):
            raise RuntimeError('Existing recovery state has no valid generation; refuse silent restart')
        return None
    failures = []
    valid = []
    for receipt_path in receipts:
        try:
            receipt = _read(receipt_path)
            assert receipt['context_sha256'] == context_sha, 'Fit context differs'
            assert receipt['execution_recovery'] == binding, 'Execution recovery binding differs'
            assert receipt['payload_path'] == Path(receipt['payload_path']).name
            payload_path = directory / receipt['payload_path']
            assert file_sha(payload_path) == receipt['payload_sha256'], 'Epoch payload SHA differs'
            payload = _torch_load(payload_path)
            assert payload['context_sha256'] == context_sha and payload['execution_recovery'] == binding
            assert object_sha(payload['fit_context']) == context_sha
            assert int(payload['epochs_completed']) == int(receipt['epochs_completed'])
            valid.append((int(payload['epochs_completed']), receipt_path.name, payload))
        except Exception as exc:
            failures.append((receipt_path.name, str(exc)))
    if not valid:
        raise RuntimeError(f'No valid epoch generation; refuse silent restart: {failures}')
    return max(valid, key=lambda item: (item[0], item[1]))[2]


def _commit_generation(directory, payload):
    epoch = int(payload['epochs_completed'])
    stem = f'epoch{epoch:04d}_{uuid.uuid4().hex}'
    payload_path = directory / (stem + '.pt')
    durable_torch(payload_path, payload)
    receipt = {'schema_version': SCHEMA, 'epochs_completed': epoch,
               'context_sha256': payload['context_sha256'],
               'execution_recovery': payload['execution_recovery'],
               'payload_path': payload_path.name, 'payload_sha256': file_sha(payload_path),
               'committed_at_unix': time.time()}
    receipt_path = directory / (stem + '.receipt.json')
    durable_json(receipt_path, receipt)
    durable_json(directory / 'LATEST.json', {'schema_version': SCHEMA,
                 'receipt_path': receipt_path.name, 'receipt_sha256': file_sha(receipt_path),
                 'epochs_completed': epoch, 'context_sha256': payload['context_sha256']})
    # Retain the two newest complete, checksum-verified generations only.
    complete = []
    for item in directory.glob('epoch*.receipt.json'):
        try:
            rec = _read(item)
            candidate = directory / rec['payload_path']
            if rec['context_sha256'] == payload['context_sha256'] and file_sha(candidate) == rec['payload_sha256']:
                complete.append((int(rec['epochs_completed']), item.name, item, candidate))
        except Exception:
            pass  # Preserve damaged evidence for review; it is never loaded.
    for _, _, rec_path, pt_path in sorted(complete, reverse=True)[2:]:
        rec_path.unlink()
        pt_path.unlink()
    _fsync_dir(directory)
    return receipt


def make_checkpoint_fit(base_module, execution_manifest_path):
    """Return the frozen fit signature with durable epoch recovery added."""
    b = base_module
    manifest_path = Path(execution_manifest_path).resolve()
    root = Path(b.ROOT).resolve()
    manifest_path.relative_to(root)
    manifest = _read(manifest_path)
    assert manifest['schema_version'] == SCHEMA and manifest['status'] == 'accepted'
    assert manifest['base_runner_sha256'] == BASE_SHA == file_sha(b.__file__)
    assert manifest['engine_sha256'] == file_sha(__file__)
    assert manifest['protocol_sha256'] == file_sha(root / 'protocol.json')
    inventory_path = _within(root, manifest.get('baseline_inventory_path', 'execution_baseline_inventory.json'))
    assert file_sha(inventory_path) == manifest['baseline_inventory_sha256']
    inventory = _read(inventory_path)
    assert inventory['schema_version'] == SCHEMA and isinstance(inventory['artifacts'], dict)
    recovery = {'manifest_sha256': file_sha(manifest_path), 'engine_sha256': file_sha(__file__),
                'adapter_sha256': manifest['adapter_sha256'], 'checkpoint_schema_version': SCHEMA}

    def fit(name, raw, y, graphs, fit_rows, early_rows, params, seed, device, path,
            epochs=None, prefetch=True, trace=False, batch_size=2048):
        path = Path(path).resolve()
        relative = str(path.relative_to(root))
        directory = path.with_suffix('.epoch_state')
        directory.mkdir(parents=True, exist_ok=True)
        _fsync_dir(directory.parent)  # Persist the new checkpoint-directory entry.
        with (directory / 'LOCK').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return _fit_locked(name, raw, y, graphs, fit_rows, early_rows, params, seed, device,
                               path, relative, directory, epochs, prefetch, trace, batch_size)

    def _fit_locked(name, raw, y, graphs, fit_rows, early_rows, params, seed, device, path,
                    relative, directory, epochs, prefetch, trace, batch_size):
        started = time.perf_counter()
        torch.manual_seed(seed)
        is_cuda = device.startswith('cuda')
        if is_cuda:
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        scaler = StandardScaler().fit(raw[fit_rows])
        X = np.ascontiguousarray(scaler.transform(raw), dtype=np.float32)
        assert np.isfinite(X).all()
        model = b.AccountNetwork(name, params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params['learning_rate']), weight_decay=1e-4)
        n = len(fit_rows)
        npos = int(y[fit_rows].sum())
        assert 0 < npos < n
        weights = np.array([n / (2 * (n - npos)), n / (2 * npos)], dtype=np.float32)
        limit = int(epochs or 200)
        optimizations = {'cache_first_mean': b.CACHE_FIRST_MEAN, 'cache_csr_blocks': b.CACHE_CSR_BLOCKS,
                         'packed_h2d': b.PACKED_H2D, 'gather_mode': b.GATHER_MODE}
        context = {'schema_version': SCHEMA, 'model_path': relative, 'protocol_sha256': manifest['protocol_sha256'],
                   'base_runner_sha256': BASE_SHA, 'name': name, 'feature_set': b.ARM, 'input_dim': b.INPUT_DIM,
                   'params': params, 'seed': seed, 'device': device,
                   'fit_rows_sha256': b.ah(fit_rows), 'early_rows_sha256': None if early_rows is None else b.ah(early_rows),
                   'fit_n': n, 'early_n': None if early_rows is None else len(early_rows),
                   'limit': limit, 'epochs_argument': epochs, 'prefetch': prefetch, 'trace': trace, 'batch_size': batch_size,
                   'raw_shape': list(raw.shape), 'raw_dtype': str(raw.dtype),
                   'scaler_hashes': {'mean': b.ah(scaler.mean_), 'scale': b.ah(scaler.scale_),
                                     'var': b.ah(scaler.var_), 'n_samples': int(scaler.n_samples_seen_)},
                   'execution_optimizations': optimizations}
        context_sha = object_sha(context)
        binding = dict(recovery, context_sha256=context_sha)
        seal_path = path.with_suffix('.completion.json')
        meta_path = path.with_suffix('.json')

        def restore_complete(metadata):
            saved = _torch_load(path)
            assert saved['metadata'] == metadata, 'Model and JSON metadata differ'
            model.load_state_dict(saved['state_dict'])
            assert b.state_digest(model) == metadata['model_digest']
            return model, X, metadata

        if seal_path.exists():
            seal = _read(seal_path)
            assert seal['execution_recovery'] == binding and seal['context_sha256'] == context_sha
            assert seal['fit_context'] == context and object_sha(seal['fit_context']) == context_sha
            assert seal['model_path'] == relative and seal['model_sha256'] == file_sha(path)
            assert seal['metadata_path'] == str(meta_path.relative_to(root)) and seal['metadata_sha256'] == file_sha(meta_path)
            metadata = _read(meta_path)
            assert metadata['execution_recovery'] == binding
            return restore_complete(metadata)
        baseline_artifacts = inventory['artifacts']
        if relative in baseline_artifacts or str(meta_path.relative_to(root)) in baseline_artifacts:
            assert file_sha(path) == baseline_artifacts[relative]
            assert file_sha(meta_path) == baseline_artifacts[str(meta_path.relative_to(root))]
            metadata = _read(meta_path)
            assert metadata['model'] == name and metadata['feature_set'] == b.ARM and metadata['input_dim'] == b.INPUT_DIM
            assert metadata['params'] == params and metadata['random_state'] == seed
            assert metadata['fit_rows_sha256'] == context['fit_rows_sha256'] and metadata['early_rows_sha256'] == context['early_rows_sha256']
            assert metadata['scaler_mean'] == scaler.mean_.tolist() and metadata['scaler_scale'] == scaler.scale_.tolist()
            assert metadata['scaler_var'] == scaler.var_.tolist() and metadata['scaler_n_samples_seen'] == int(scaler.n_samples_seen_)
            assert metadata['fit_n'] == n and metadata['fit_positive'] == npos and metadata['class_weights'] == weights.tolist()
            assert metadata['seed_only_loss'] is True and (metadata['qa_trace'] is not None) == trace
            assert metadata['sampling_prefetch'] == prefetch and metadata['seed_batch_size'] == batch_size
            assert metadata['execution_optimizations'] == optimizations
            if early_rows is None:
                assert metadata['epochs_run'] == limit
            else:
                assert metadata['epochs_run'] <= limit
            return restore_complete(metadata)

        gpuX = torch.from_numpy(X).to(device)
        gpuY = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).to(device)
        gpuW = torch.from_numpy(weights).to(device)
        rng = np.random.default_rng(seed + 1000000)
        best_auc = -np.inf
        best_epoch = 0
        best_state = None
        history = []
        totals = {'sampling_cpu_seconds': 0., 'sampling_wait_seconds': 0., 'cuda_transfer_seconds': 0.,
                  'cuda_forward_backward_seconds': 0., 'early_evaluation_seconds': 0., 'checkpoint_io_seconds': 0.,
                  'recovery_checkpoint_io_seconds': 0.}
        qa_trace = {'sampling_sha256': [], 'batch_loss': []} if trace else None
        previous = _load_generation(directory, context_sha, binding)
        start_epoch = 1
        terminal = False
        if previous is not None:
            model.load_state_dict(previous['model_state_dict'])
            optimizer.load_state_dict(previous['optimizer_state_dict'])
            best_auc = previous['best_auc']
            best_epoch = previous['best_epoch']
            best_state = previous['best_state']
            history = previous['history']
            totals = previous['totals']
            qa_trace = previous['qa_trace']
            start_epoch = int(previous['epochs_completed']) + 1
            terminal = previous['terminal']
            started = time.perf_counter() - previous['active_seconds']
            _restore_rng(previous['rng_state'], rng, is_cuda)
        else:
            if path.exists() or meta_path.exists():
                raise RuntimeError('Unsealed completed fit is not in baseline inventory and has no legal epoch checkpoint')

        def commit(epoch, terminal):
            if is_cuda:
                torch.cuda.synchronize()
            tick = time.perf_counter()
            payload = {'schema_version': SCHEMA, 'fit_context': context, 'context_sha256': context_sha,
                       'execution_recovery': binding, 'epochs_completed': epoch, 'terminal': terminal,
                       'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                       'best_auc': best_auc, 'best_epoch': best_epoch, 'best_state': best_state,
                       'history': history, 'totals': dict(totals), 'qa_trace': qa_trace,
                       'rng_state': _rng_state(rng, is_cuda), 'active_seconds': time.perf_counter() - started}
            receipt = _commit_generation(directory, payload)
            totals['recovery_checkpoint_io_seconds'] += time.perf_counter() - tick
            if EPOCH_COMMITTED_HOOK is not None:
                EPOCH_COMMITTED_HOOK(path, epoch, receipt)

        if previous is None:
            commit(0, False)
            # The marker follows epoch zero. A power loss during marker writing
            # therefore still leaves an independently verifiable initial state.
            durable_json(directory / 'fit_context.json', {'fit_context': context, 'context_sha256': context_sha,
                                                        'execution_recovery': binding})
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='label_blind_sampler') as sampler:
            for epoch in range(start_epoch, limit + 1) if not terminal else ():
                epoch_start = time.perf_counter()
                model.train()
                total_loss = torch.zeros((), device=device)
                ordered = rng.permutation(fit_rows)
                cursor = 0
                future = sampler.submit(b.sampled_batch, graphs, ordered[:batch_size], rng)
                cursor = batch_size
                events = []
                ep_sampling = ep_wait = 0.
                while future is not None:
                    tick = time.perf_counter()
                    roots, second, r1, m1, r0, m0, sampling_seconds = future.result()
                    ep_wait += time.perf_counter() - tick
                    ep_sampling += sampling_seconds
                    has_next = cursor < n
                    next_roots = ordered[cursor:cursor + batch_size]
                    cursor += batch_size
                    future = sampler.submit(b.sampled_batch, graphs, next_roots, rng) if has_next and prefetch else None
                    if trace:
                        digest = hashlib.sha256()
                        for value in (roots, second, r1, r0):
                            digest.update(np.ascontiguousarray(value).tobytes())
                        for maps in (m1, m0):
                            for position, valid in maps:
                                digest.update(np.ascontiguousarray(position).tobytes())
                                digest.update(np.ascontiguousarray(valid).tobytes())
                        qa_trace['sampling_sha256'].append(digest.hexdigest())
                    if is_cuda:
                        begin = torch.cuda.Event(enable_timing=True)
                        transferred = torch.cuda.Event(enable_timing=True)
                        trained = torch.cuda.Event(enable_timing=True)
                        begin.record()
                    if is_cuda and b.PACKED_H2D:
                        second_t, root_t, inner, outer = b.packed_transfer(second, roots, r1, m1, r0, m0, device)
                    else:
                        second_t = torch.as_tensor(second, dtype=torch.long, device=device)
                        root_t = torch.as_tensor(roots, dtype=torch.long, device=device)
                        inner = b.tensor_block(r1, m1, device)
                        outer = b.tensor_block(r0, m0, device)
                    if is_cuda:
                        transferred.record()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model.forward_sampled(gpuX[second_t], inner, outer)
                    labels = gpuY[root_t]
                    w = gpuW[labels.long()]
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none') * w).mean()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach() * len(roots)
                    if b.STEP_OBSERVER is not None:
                        b.STEP_OBSERVER(model, optimizer, epoch, roots)
                    if is_cuda:
                        trained.record()
                        events.append((begin, transferred, trained))
                    if trace:
                        qa_trace['batch_loss'].append(float(loss.detach().cpu()))
                    if has_next and not prefetch:
                        future = sampler.submit(b.sampled_batch, graphs, next_roots, rng)
                if is_cuda:
                    torch.cuda.synchronize()
                    transfer = sum(a.elapsed_time(b_) for a, b_, c in events) / 1000
                    gpu_train = sum(b_.elapsed_time(c) for a, b_, c in events) / 1000
                else:
                    transfer = gpu_train = 0.
                record = {'epoch': epoch, 'weighted_seed_loss': float(total_loss.cpu()) / n,
                          'seconds': time.perf_counter() - started, 'epoch_fit_seconds': time.perf_counter() - epoch_start,
                          'sampling_cpu_seconds': ep_sampling, 'sampling_wait_seconds': ep_wait,
                          'cuda_transfer_seconds': transfer, 'cuda_forward_backward_seconds': gpu_train}
                totals['sampling_cpu_seconds'] += ep_sampling
                totals['sampling_wait_seconds'] += ep_wait
                totals['cuda_transfer_seconds'] += transfer
                totals['cuda_forward_backward_seconds'] += gpu_train
                stopped = False
                if early_rows is not None:
                    tick = time.perf_counter()
                    score = b.predict(model, X, graphs, early_rows)
                    auc = float(b.roc_auc_score(y[early_rows], score))
                    record['early_auroc'] = auc
                    record['early_evaluation_seconds'] = time.perf_counter() - tick
                    totals['early_evaluation_seconds'] += record['early_evaluation_seconds']
                    if auc > best_auc:
                        best_auc = auc
                        best_epoch = epoch
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stopped = epoch - best_epoch >= 30
                else:
                    best_epoch = epoch
                history.append(record)
                # All sampler futures are consumed, CUDA drained, early scoring done.
                commit(epoch, stopped or epoch == limit)
                if stopped:
                    break
                if epoch % 10 == 0 or epoch == 1:
                    print(json.dumps({'event': 'fit_epoch', 'path': relative, 'model': name, 'epoch': epoch,
                                      **record, **b.memory()}), flush=True)
        if early_rows is not None:
            model.load_state_dict(best_state)
        if is_cuda:
            torch.cuda.synchronize()
        metadata = {'model': name, 'feature_set': b.ARM, 'input_dim': b.INPUT_DIM, 'params': params,
                    'device': device, 'random_state': seed, 'scaler_fit_rows_sha256': b.ah(fit_rows),
                    'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                    'scaler_var': scaler.var_.tolist(), 'scaler_n_samples_seen': int(scaler.n_samples_seen_),
                    'fit_rows_sha256': b.ah(fit_rows), 'early_rows_sha256': None if early_rows is None else b.ah(early_rows),
                    'fit_n': n, 'fit_positive': npos, 'class_weights': weights.tolist(), 'seed_only_loss': True,
                    'selected_epochs': best_epoch, 'epochs_run': len(history),
                    'best_early_auroc': None if early_rows is None else best_auc, 'history': history,
                    'model_digest': b.state_digest(model), 'seconds': time.perf_counter() - started,
                    'phase_timing': totals, 'qa_trace': qa_trace, 'sampling_prefetch': prefetch,
                    'seed_batch_size': batch_size, 'execution_optimizations': optimizations,
                    'execution_recovery': binding,
                    'timing_note': 'CPU sampling overlaps CUDA fit; phase times are not additive. CUDA events synchronized once per epoch. Recovery retains active execution time, excluding downtime.',
                    **b.memory()}
        tick = time.perf_counter()
        durable_torch(path, {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'metadata': metadata})
        totals['checkpoint_io_seconds'] = time.perf_counter() - tick
        metadata['seconds'] = time.perf_counter() - started
        durable_torch(path, {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'metadata': metadata})
        durable_json(meta_path, metadata)
        durable_json(seal_path, {'schema_version': SCHEMA, 'fit_context': context, 'context_sha256': context_sha,
                                'execution_recovery': binding, 'model_path': relative, 'model_sha256': file_sha(path),
                                'metadata_path': str(meta_path.relative_to(root)), 'metadata_sha256': file_sha(meta_path),
                                'completed_at_unix': time.time()})
        del gpuX, gpuY, gpuW
        return model, X, metadata

    fit.__name__ = 'fit'
    return fit
