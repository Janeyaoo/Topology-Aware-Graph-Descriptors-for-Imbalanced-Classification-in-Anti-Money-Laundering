"""Evaluate frozen scores at fixed, per-outer-fold alert budgets; never fit models.

The manifest fixes 42 settings, 1,050 predictions, canonical accounts/splits and
74 directed dataset-level contrasts. Completed computations remain provisional
until a separate verifier accepts them. No significance tests are calculated.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

SCRIPT = Path(__file__).resolve()
VERSION = 1
AP_TOLERANCE = 1e-10
BUDGETS = (.005, .01, .02)
MODELS = ('lightgbm', 'xgboost', 'random_forest', 'mlp', 'graphsage_weak')
ARMS = ('base16', 'base_g32', 'gmke44', 'gmkep56')
BUDGET_KEYS = dict(zip(BUDGETS, ('selected_row_index_b005', 'selected_row_index_b010', 'selected_row_index_b020')))
IDENTITY = ['dataset', 'model', 'feature_set', 'seed', 'fold']
SETTING = ['dataset', 'model', 'feature_set']
FOLD_COLUMNS = IDENTITY + ['budget', 'k', 'n_test', 'n_positive', 'tp', 'precision_at_k', 'recall_at_k']
SUMMARY_COLUMNS = SETTING + ['budget', 'precision_mean', 'precision_sd', 'recall_mean', 'recall_sd']
SEED_COLUMNS = SETTING + ['seed', 'budget', 'k', 'n_test', 'n_positive', 'tp', 'precision_at_k', 'recall_at_k', 'n_folds']
SELECT_COLUMNS = IDENTITY + ['budget', 'k', 'selected_path', 'selected_sha256', 'selected_array_key']
PAIR_IDENTITY = ['dataset', 'pair_id', 'left_model', 'left_feature_set', 'right_model', 'right_feature_set']
PAIR_FOLD_COLUMNS = PAIR_IDENTITY + ['seed', 'fold', 'budget', 'k', 'n_test', 'n_positive', 'tp_left', 'tp_right', 'delta_tp', 'delta_precision_at_k', 'delta_recall_at_k']
PAIR_SEED_COLUMNS = PAIR_IDENTITY + ['seed', 'budget', 'k', 'n_test', 'n_positive', 'tp_left', 'tp_right', 'delta_tp', 'delta_precision_at_k', 'delta_recall_at_k', 'n_folds']
PAIR_SUMMARY_COLUMNS = PAIR_IDENTITY + ['budget', 'n_folds', 'n_seeds', 'precision_delta_mean', 'precision_delta_sd', 'recall_delta_mean', 'recall_delta_sd'] + [
    f'{metric}_{direction}_{unit}' for unit in ('folds', 'seeds') for metric in ('precision', 'recall') for direction in ('positive', 'zero', 'negative')]

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def require(condition, message):
    if not condition: raise ValueError(message)
def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()
def ah(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def oh(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def sync_dir(directory):
    fd = os.open(directory, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
def atomic_bytes(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path); sync_dir(path.parent)
def save(path, value): atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
def csv(path, frame): atomic_bytes(path, frame.to_csv(index=False).encode())
def npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path); sync_dir(path.parent)
def segment(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) is not None, 'Unsafe identity path component')
    return value
def identity(entry): return tuple(entry[key] for key in IDENTITY)
def setting(entry): return tuple(entry[key] for key in SETTING)
def entry_id(entry): return '/'.join(str(value) for value in identity(entry))
def resolved(base, value):
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()
def descriptor(base, record):
    require(isinstance(record, dict) and 'path' in record and 'sha256' in record, 'Input descriptor requires path and SHA256')
    path = resolved(base, record['path'])
    require(isinstance(record['sha256'], str) and re.fullmatch('[0-9a-f]{64}', record['sha256']) is not None, 'Invalid SHA256')
    return path, record['sha256']
def verify_file(path, expected):
    require(path.is_file(), f'Missing input: {path}')
    require(sha(path) == expected, f'Input SHA256 mismatch: {path}')
def protocol_descriptor(manifest):
    if 'protocol' in manifest: return manifest['protocol']
    return {'path': manifest['protocol_path'], 'sha256': manifest.get('protocol_sha256', manifest.get('protocol_hash'))}

def expand_pairs(manifest):
    result = []
    for record in manifest['pairs']:
        datasets = record.get('datasets', [record.get('dataset')])
        for ds in datasets:
            pair_id = record.get('pair_id', record.get('id'))
            segment(pair_id); segment(ds)
            left, right = record['left'], record['right']
            result.append(dict(dataset=ds, pair_id=pair_id, left_model=left['model'], left_feature_set=left['feature_set'],
                               right_model=right['model'], right_feature_set=right['feature_set']))
    require(len(result) == 74 and len({tuple(row[key] for key in ('dataset', 'pair_id')) for row in result}) == 74,
            'Exactly 74 unique directed dataset-level pairs must be preregistered')
    return sorted(result, key=lambda row: (row['dataset'], row['pair_id']))

def validate_manifest(manifest):
    require(manifest.get('schema_version') == 1, 'Unsupported manifest schema')
    entries = manifest['entries']; require(len(entries) == 1050, 'Manifest must enumerate all 1,050 planned outer-fold scores')
    keys = set(); groups = {}
    for entry in entries:
        for key in SETTING: segment(entry[key])
        for key in ('seed', 'fold'): require(type(entry[key]) is int and 0 <= entry[key] < 5, 'Seed/fold outside frozen scope')
        require(identity(entry) not in keys, 'Duplicate prediction identity'); keys.add(identity(entry))
        require('path' in entry or 'prediction_path' in entry, 'Prediction path missing')
        ap = entry['expected_ap']; require(isinstance(ap, (int, float)) and math.isfinite(ap) and 0 <= ap <= 1, 'Invalid formal AP')
        require(re.fullmatch('[0-9a-f]{64}', entry['sha256']) is not None, 'Invalid prediction SHA256')
        groups.setdefault(setting(entry), set()).add((entry['seed'], entry['fold']))
    require(len(groups) == 42 and all(value == {(s, f) for s in range(5) for f in range(5)} for value in groups.values()), 'All 42 settings need the same 25 outer folds')
    require({key[0] for key in groups} == {'hi_small', 'li_small'} and all(sum(key[0] == ds for key in groups) == 21 for ds in ('hi_small', 'li_small')), 'Two datasets with 21 settings each required')
    expected_settings = {(ds, model, arm) for ds in ('hi_small', 'li_small') for model in MODELS for arm in ARMS}
    expected_settings.update((ds, 'xgboost', 'base_gfp_account_stream') for ds in ('hi_small', 'li_small'))
    require(set(groups) == expected_settings, 'Registered model/representation inventory differs from the fixed 42 settings')
    pairs = expand_pairs(manifest)
    for pair in pairs:
        for side in ('left', 'right'):
            require((pair['dataset'], pair[side + '_model'], pair[side + '_feature_set']) in groups, 'Pair references an unregistered setting')
    expected_pairs = set()
    for ds in ('hi_small', 'li_small'):
        for model in MODELS:
            expected_pairs.update((ds, model, ARMS[left], model, ARMS[right]) for left in range(1, 4) for right in range(left))
        expected_pairs.add((ds, 'xgboost', 'base_gfp_account_stream', 'xgboost', 'gmkep56'))
        expected_pairs.update((ds, 'graphsage_weak', arm, 'mlp', arm) for arm in ARMS)
        expected_pairs.add((ds, 'graphsage_weak', 'base16', 'xgboost', 'gmkep56'))
        expected_pairs.add((ds, 'mlp', 'gmkep56', 'xgboost', 'gmkep56'))
    require({tuple(pair[key] for key in ('dataset', 'left_model', 'left_feature_set', 'right_model', 'right_feature_set')) for pair in pairs} == expected_pairs,
            'Registered contrasts differ from all six within-model representation pairs and the fourteen cross-method pairs')
    require('legacy_budget' in manifest and 'legacy_budget_summary' in manifest and 'canonical' in manifest,
            'Canonical accounts/splits and both legacy tables must be bound')
    protocol_descriptor(manifest)
    if 'budgets' in manifest: require(tuple(manifest['budgets']) == BUDGETS, 'Alert budgets are frozen')
    return sorted(entries, key=identity), pairs

class Canonical:
    """Keep one account ledger and its small split arrays resident at a time."""
    def __init__(self, manifest, base):
        self.manifest = manifest; self.base = base; self.dataset = None; self.sources = {}
    def load(self, ds):
        if self.dataset == ds: return
        record = self.manifest['canonical'][ds]
        path, expected = descriptor(self.base, record['accounts']); verify_file(path, expected)
        row_col = record['accounts'].get('row_index_col', 'row_index')
        id_col = record['accounts'].get('account_id_col', 'account_id')
        label_col = record['accounts'].get('label_col', 'label')
        frame = pd.read_parquet(path, columns=[row_col, id_col, label_col]).sort_values(row_col, kind='mergesort')
        rows = frame[row_col].to_numpy(); labels = frame[label_col].to_numpy()
        require(np.issubdtype(rows.dtype, np.integer), 'Canonical row_index must remain integer')
        np.testing.assert_array_equal(rows, np.arange(len(rows), dtype=np.int64))
        require(frame[id_col].is_unique and not frame[id_col].isna().any(), 'Canonical account IDs must be unique')
        require(set(np.unique(labels)) == {0, 1}, 'Canonical ledger must contain both labels')
        self.labels = np.ascontiguousarray(labels, dtype=np.int8)
        self.account_ids = frame[id_col].to_numpy(); self.ledger_sha = expected
        self.sources[str(path)] = expected
        splits = record['splits']; mapping = {}
        if isinstance(splits, dict):
            records = []
            for key, value in splits.items():
                match = re.fullmatch(r'seed([0-4])_fold([0-4])', key)
                require(match is not None, 'Canonical split mapping key must be seedS_foldF')
                records.append({**value, 'seed': int(match[1]), 'fold': int(match[2])})
        else: records = splits
        require(len(records) == 25, 'Canonical dataset requires 25 frozen splits')
        for split in records:
            key = (split['seed'], split['fold']); require(key not in mapping, 'Duplicate canonical split')
            file, digest = descriptor(self.base, split); verify_file(file, digest)
            with np.load(file, allow_pickle=False) as data:
                test = data[split.get('outer_test_key', 'outer_test')].copy()
            require(np.issubdtype(test.dtype, np.integer) and len(np.unique(test)) == len(test), 'Invalid canonical outer_test')
            require((test >= 0).all() and (test < len(rows)).all(), 'Canonical test row out of range')
            mapping[key] = (np.sort(test), digest); self.sources[str(file)] = digest
        require(set(mapping) == {(s, f) for s in range(5) for f in range(5)}, 'Canonical seed/fold grid incomplete')
        for seed in range(5):
            np.testing.assert_array_equal(np.sort(np.concatenate([mapping[seed, fold][0] for fold in range(5)])), rows)
        self.splits = mapping; self.dataset = ds
    def expected(self, entry):
        self.load(entry['dataset']); rows, split_hash = self.splits[entry['seed'], entry['fold']]
        return rows, self.labels[rows], self.account_ids[rows], dict(accounts_sha256=self.ledger_sha, split_sha256=split_hash,
            row_index_sha256=ah(rows), labels_sha256=ah(self.labels[rows]))

def ranked_indices(rows, probability):
    """Sort account rows first; stable descending scores resolve ties by row_index."""
    row_order = np.argsort(rows, kind='mergesort')
    rows = rows[row_order]; probability = probability[row_order]
    rank = np.argsort(-probability, kind='mergesort')
    return row_order, rank

def process_entry(entry, base, output, manifest_hash, canonical):
    source = resolved(base, entry.get('path', entry.get('prediction_path')))
    verify_file(source, entry['sha256'])
    expected_rows, expected_labels, expected_ids, binding = canonical.expected(entry)
    directory = output / 'cache' / entry['dataset'] / entry['model'] / entry['feature_set'] / f"seed{entry['seed']}_fold{entry['fold']}"
    record_path = directory / 'record.json'; selected_path = directory / 'selected.npz'
    receipt_path = directory / 'receipt.json'
    seal = dict(manifest_sha256=manifest_hash, runner_sha256=sha(SCRIPT), entry_sha256=oh(entry),
                prediction_sha256=entry['sha256'], canonical_binding=binding)
    if receipt_path.exists():
        require(record_path.is_file(), f'Committed cache record missing: {record_path}')
        receipt = read(receipt_path)
        require(receipt['binding'] == seal and receipt['record_sha256'] == sha(record_path), 'Cached record digest/binding mismatch')
        saved = read(record_path)
        require(saved['binding'] == seal, f'Cached result binding changed: {record_path}')
        verify_file(selected_path, saved['selected_sha256'])
        require(receipt['selected_sha256'] == saved['selected_sha256'], 'Cache receipt/selection digest mismatch')
        require(saved['entry_identity'] == list(identity(entry)) and len(saved['fold_values']) == 3, 'Invalid cached identity/budget coverage')
        return saved
    row_col = entry.get('row_index_col', 'row_index')
    label_col = entry.get('label_col', entry.get('y_col', 'y_true'))
    score_col = entry.get('score_col', 'y_score')
    id_col = entry.get('account_id_col', 'account_id')
    columns = [row_col, label_col, score_col]
    has_ids = id_col in pq.read_schema(source).names
    if has_ids: columns.append(id_col)
    frame = pd.read_parquet(source, columns=columns)
    require(len(frame) == len(expected_rows), 'Prediction fold size differs from canonical outer_test')
    rows = frame[row_col].to_numpy(); labels = frame[label_col].to_numpy(); probability = frame[score_col].to_numpy()
    require(np.issubdtype(rows.dtype, np.integer) and len(np.unique(rows)) == len(rows), 'Duplicate/non-integer prediction row_index')
    require(np.issubdtype(probability.dtype, np.floating), 'Continuous predictions must retain their stored floating dtype')
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), 'Invalid probability')
    order, rank = ranked_indices(rows, probability)
    rows, labels, probability = rows[order], labels[order], probability[order]
    np.testing.assert_array_equal(rows, expected_rows)
    np.testing.assert_array_equal(labels, expected_labels)
    if has_ids: np.testing.assert_array_equal(frame[id_col].to_numpy()[order], expected_ids)
    require(int(labels.sum()) > 0, 'AP/recall require positive canonical test accounts')
    ap = float(average_precision_score(labels, probability))
    require(abs(ap - entry['expected_ap']) <= AP_TOLERANCE, f'Formal AP mismatch: {entry_id(entry)} ({ap} vs {entry["expected_ap"]})')
    n = len(rows); positives = int(labels.sum()); values = []; arrays = {}
    for budget in BUDGETS:
        k = int(math.ceil(budget * n)); selected = rank[:k]
        tp = int(labels[selected].sum()); key = BUDGET_KEYS[budget]
        arrays[key] = np.ascontiguousarray(rows[selected], dtype=np.int64)
        values.append({**{key: entry[key] for key in IDENTITY}, 'budget': budget, 'k': k, 'n_test': n,
            'n_positive': positives, 'tp': tp, 'precision_at_k': tp / k, 'recall_at_k': tp / positives})
    npz(selected_path, **arrays, entry_id=np.array(entry_id(entry)), budgets=np.array(BUDGETS, dtype=np.float64))
    saved = dict(entry_identity=list(identity(entry)), binding=seal, prediction_path=str(source),
        computed_ap=ap, expected_ap=entry['expected_ap'], ap_absolute_difference=abs(ap - entry['expected_ap']),
        score_dtype=str(probability.dtype), score_sha256_in_canonical_row_order=ah(probability),
        account_ids_checked=has_ids, selected_path=str(selected_path), selected_sha256=sha(selected_path),
        fold_values=values, completed_at=now())
    save(record_path, saved)
    save(receipt_path, dict(binding=seal, record_sha256=sha(record_path), selected_sha256=saved['selected_sha256'], committed_at=now()))
    return saved

def make_summary(folds):
    rows = []
    for key, group in folds.groupby(SETTING + ['budget'], sort=True):
        if len(group) != 25: continue
        rows.append(dict(zip(SETTING + ['budget'], key), precision_mean=float(group.precision_at_k.mean()),
            precision_sd=float(group.precision_at_k.std(ddof=1)), recall_mean=float(group.recall_at_k.mean()),
            recall_sd=float(group.recall_at_k.std(ddof=1))))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
def make_seeds(folds):
    rows = []
    for key, group in folds.groupby(SETTING + ['seed', 'budget'], sort=True):
        if len(group) != 5: continue
        counts = {column: int(group[column].sum()) for column in ('k', 'n_test', 'n_positive', 'tp')}
        rows.append({**dict(zip(SETTING + ['seed', 'budget'], key)), **counts,
            'precision_at_k': counts['tp'] / counts['k'], 'recall_at_k': counts['tp'] / counts['n_positive'], 'n_folds': 5})
    return pd.DataFrame(rows, columns=SEED_COLUMNS)
def pairs_tables(folds, seeds, registered_pairs):
    fold_rows = []; seed_rows = []; summaries = []
    for pair in registered_pairs:
        def compare(frame, seed_mode=False):
            keys = ['seed', 'budget'] if seed_mode else ['seed', 'fold', 'budget']
            def select(side):
                return frame[(frame.dataset == pair['dataset']) & (frame.model == pair[side + '_model']) &
                    (frame.feature_set == pair[side + '_feature_set'])].set_index(keys)
            left, right = select('left'), select('right'); common = left.index.intersection(right.index).sort_values()
            result = []
            for index in common:
                a, b = left.loc[index], right.loc[index]
                require(all(int(a[column]) == int(b[column]) for column in ('k', 'n_test', 'n_positive')), 'Paired fold/seed account or budget mismatch')
                result.append({**pair, **dict(zip(keys, index)), **{column: int(a[column]) for column in ('k', 'n_test', 'n_positive')},
                    'tp_left': int(a.tp), 'tp_right': int(b.tp), 'delta_tp': int(a.tp) - int(b.tp),
                    'delta_precision_at_k': float(a.precision_at_k - b.precision_at_k),
                    'delta_recall_at_k': float(a.recall_at_k - b.recall_at_k), **({'n_folds': 5} if seed_mode else {})})
            return result
        fold_values = compare(folds); seed_values = compare(seeds, True)
        fold_rows.extend(fold_values); seed_rows.extend(seed_values)
        for budget in BUDGETS:
            fs = [row for row in fold_values if row['budget'] == budget]
            ss = [row for row in seed_values if row['budget'] == budget]
            if len(fs) != 25 or len(ss) != 5: continue
            value = {**pair, 'budget': budget, 'n_folds': 25, 'n_seeds': 5}
            for metric in ('precision', 'recall'):
                d = np.array([row[f'delta_{metric}_at_k'] for row in fs], dtype=np.float64)
                value[f'{metric}_delta_mean'] = float(d.mean()); value[f'{metric}_delta_sd'] = float(d.std(ddof=1))
                for unit, records in (('folds', fs), ('seeds', ss)):
                    increments = np.array([row[f'delta_{metric}_at_k'] for row in records])
                    for direction, mask in (('positive', increments > 0), ('zero', increments == 0), ('negative', increments < 0)):
                        value[f'{metric}_{direction}_{unit}'] = int(mask.sum())
            summaries.append(value)
    return (pd.DataFrame(fold_rows, columns=PAIR_FOLD_COLUMNS), pd.DataFrame(seed_rows, columns=PAIR_SEED_COLUMNS),
            pd.DataFrame(summaries, columns=PAIR_SUMMARY_COLUMNS))

def legacy_check(folds, descriptor_record, base, complete):
    path, digest = descriptor(base, descriptor_record); verify_file(path, digest)
    old = pd.read_csv(path, float_precision='round_trip')
    require(list(old.columns) == FOLD_COLUMNS and len(old) == 1350, 'Legacy 1,350-row table schema/scope changed')
    keys = IDENTITY + ['budget']; merged = old.merge(folds, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    for column in ('k', 'n_test', 'n_positive', 'tp'):
        np.testing.assert_array_equal(merged[column + '_old'].to_numpy(), merged[column + '_new'].to_numpy())
    max_difference = 0.
    for column in ('precision_at_k', 'recall_at_k'):
        differences = np.abs(merged[column + '_old'].to_numpy() - merged[column + '_new'].to_numpy())
        maximum = float(differences.max()) if len(differences) else 0.; max_difference = max(max_difference, maximum)
        require(maximum <= 1e-15, 'Legacy budget result failed exact account/tie reproduction')
    if complete: require(len(merged) == 1350, 'Full run must reproduce all legacy 1,350 budget rows')
    return dict(path=str(path), sha256=digest, expected_rows=1350, compared_rows=len(merged),
        integer_counts_exact=True, max_absolute_precision_recall_difference=max_difference)
def legacy_summary_check(summary, descriptor_record, base, complete):
    path, digest = descriptor(base, descriptor_record); verify_file(path, digest)
    old = pd.read_csv(path, float_precision='round_trip')
    require(list(old.columns) == SUMMARY_COLUMNS and len(old) == 54, 'Legacy 54-row summary schema/scope changed')
    keys = SETTING + ['budget']; merged = old.merge(summary, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    maximum = 0.
    for column in ('precision_mean', 'precision_sd', 'recall_mean', 'recall_sd'):
        differences = np.abs(merged[column + '_old'].to_numpy() - merged[column + '_new'].to_numpy())
        value = float(differences.max()) if len(differences) else 0.; maximum = max(maximum, value)
        require(value <= 1e-15, 'Legacy mean/sample-SD reproduction failed')
    if complete: require(len(merged) == 54, 'Full run must reproduce all 54 legacy summaries')
    return dict(path=str(path), sha256=digest, expected_rows=54, compared_rows=len(merged), maximum_absolute_difference=maximum)

def run(args):
    manifest_path = args.manifest.resolve(); base = manifest_path.parent; manifest = read(manifest_path)
    entries, registered_pairs = validate_manifest(manifest); manifest_hash = sha(manifest_path)
    protocol_path, protocol_hash = descriptor(base, protocol_descriptor(manifest)); verify_file(protocol_path, protocol_hash)
    legacy_path, legacy_hash = descriptor(base, manifest['legacy_budget']); verify_file(legacy_path, legacy_hash)
    old_summary_path, old_summary_hash = descriptor(base, manifest['legacy_budget_summary']); verify_file(old_summary_path, old_summary_hash)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    with (output / 'analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (output / 'COMPLETE.json').exists(), 'Independent acceptance already sealed this directory; do not rewrite it')
        expected = dict(schema_version=VERSION, manifest_path=str(manifest_path), manifest_sha256=manifest_hash,
            runner_sha256=sha(SCRIPT), expected_entries=1050, expected_settings=42, expected_dataset_pairs=74,
            budgets=list(BUDGETS), ranking='ascending integer row_index first, stable descending original stored scores; ceil(b*n) independently per outer fold',
            seed_aggregation='sum the five disjoint fold-selected TP/alerts/positives; no across-fold score ranking',
            no_fitting=True, no_significance_tests=True)
        expected.update(protocol_sha256=protocol_hash, legacy_budget_sha256=legacy_hash,
                        legacy_budget_summary_sha256=old_summary_hash, ap_tolerance=AP_TOLERANCE)
        registration = output / 'analysis_registration.json'
        if registration.exists():
            previous = read(registration)
            require(all(previous.get(key) == value for key, value in expected.items()), 'Analysis registration/runner/manifest changed')
            verify_file(output / 'manifest_snapshot.json', manifest_hash)
            verify_file(output / 'runner_snapshot.py', expected['runner_sha256'])
        else:
            shutil.copy2(manifest_path, output / 'manifest_snapshot.json'); shutil.copy2(SCRIPT, output / 'runner_snapshot.py')
            save(registration, {**expected, 'registered_at': now()})
        canonical = Canonical(manifest, base); records = []; pending = []
        progress_path = output / 'progress.json'
        try:
            for index, entry in enumerate(entries):
                if index % 10 == 0:
                    require(shutil.disk_usage(output).free >= 6 * 1024 ** 3, 'Less than preregistered 6 GiB local free-space reserve')
                source = resolved(base, entry.get('path', entry.get('prediction_path')))
                if not source.is_file():
                    pending.append(dict(entry_id=entry_id(entry), path=str(source)))
                else:
                    records.append(process_entry(entry, base, output, manifest_hash, canonical))
                if index % 10 == 0 or index + 1 == len(entries):
                    save(progress_path, dict(status='COMPUTING_PROVISIONAL', updated_at=now(), manifest_sha256=manifest_hash,
                        inspected_entries=index + 1, cached_accepted_entries=len(records), missing_entries=len(pending), expected_entries=1050))
            complete = len(records) == 1050
            folds = pd.DataFrame([row for record in records for row in record['fold_values']], columns=FOLD_COLUMNS)
            if len(folds): folds = folds.sort_values(IDENTITY + ['budget'], kind='mergesort').reset_index(drop=True)
            selections = pd.DataFrame([{**{key: row[key] for key in IDENTITY}, 'budget': row['budget'], 'k': row['k'],
                'selected_path': record['selected_path'], 'selected_sha256': record['selected_sha256'],
                'selected_array_key': BUDGET_KEYS[row['budget']]} for record in records for row in record['fold_values']], columns=SELECT_COLUMNS)
            summary = make_summary(folds); seeds = make_seeds(folds)
            paired_folds, paired_seeds, paired_summary = pairs_tables(folds, seeds, registered_pairs)
            old = legacy_check(folds, manifest['legacy_budget'], base, complete)
            old_summary = legacy_summary_check(summary, manifest['legacy_budget_summary'], base, complete)
            if complete:
                require((len(folds), len(summary), len(seeds), len(paired_folds), len(paired_seeds), len(paired_summary)) ==
                    (3150, 126, 630, 5550, 1110, 222), 'Full unified output scope incomplete')
            destination = output if complete else output / 'partial'
            files = {'budget_fold_values.csv': folds, 'budget_summary.csv': summary, 'budget_seed_values.csv': seeds,
                'budget_paired_fold_values.csv': paired_folds, 'budget_paired_seed_values.csv': paired_seeds,
                'budget_paired_summary.csv': paired_summary, 'selection_manifest.csv': selections}
            operating = folds[IDENTITY + ['budget']].copy()
            operating['fp'] = folds.k - folds.tp; operating['actual_fraction'] = folds.k / folds.n_test
            files['budget_operating_counts.csv'] = operating
            coverage = []
            for key, group in folds.groupby(SETTING, sort=True):
                coverage.append({**dict(zip(SETTING, key)), 'completed_outer_folds': len(group) // 3, 'expected_outer_folds': 25})
            files['coverage.csv'] = pd.DataFrame(coverage, columns=SETTING + ['completed_outer_folds', 'expected_outer_folds'])
            for filename, frame in files.items(): csv(destination / filename, frame)
            output_hashes = {str(destination / filename): sha(destination / filename) for filename in files}
            output_hashes.update({record['selected_path']: record['selected_sha256'] for record in records})
            sources = {str(resolved(base, entry.get('path', entry.get('prediction_path')))): entry['sha256'] for entry in entries
                       if resolved(base, entry.get('path', entry.get('prediction_path'))).is_file()}
            sources.update(canonical.sources); sources[old['path']] = old['sha256']
            sources.update({old_summary['path']: old_summary['sha256'], str(protocol_path): protocol_hash})
            report = dict(status='READY_FOR_INDEPENDENT_VALIDATION' if complete else 'PARTIAL_NOT_COMPLETE',
                producer_checks_passed=True, independent_acceptance=False, created_at=now(), manifest_sha256=manifest_hash,
                runner_sha256=sha(SCRIPT), registration_sha256=sha(registration), expected_entries=1050,
                accepted_entries=len(records), missing_entries=pending, complete_settings=len(summary) // 3,
                fold_rows=len(folds), summary_rows=len(summary), seed_rows=len(seeds), paired_fold_rows=len(paired_folds),
                paired_seed_rows=len(paired_seeds), paired_summary_rows=len(paired_summary), legacy_reproduction=old,
                legacy_summary_reproduction=old_summary, ap_tolerance=AP_TOLERANCE,
                max_ap_difference=max((record['ap_absolute_difference'] for record in records), default=0.),
                source_sha256=sources, output_sha256=output_hashes,
                note='AP and label/row checks validate frozen score inputs; all budget reports remain provisional until independent acceptance. Descriptive paired increments only.')
            save(destination / 'analysis_result.json', report); save(progress_path, {key: value for key, value in report.items() if key not in ('source_sha256', 'output_sha256')})
            print(json.dumps({key: value for key, value in report.items() if key in ('status', 'accepted_entries', 'complete_settings', 'fold_rows', 'summary_rows', 'seed_rows', 'paired_summary_rows')}), flush=True)
            if not complete and not args.allow_incomplete: raise RuntimeError('Predictions missing; partial outputs retained. Use --allow-incomplete only for explicit partial processing.')
            return 0
        except BaseException as error:
            save(output / 'last_failure.json', dict(status='FAILED_NOT_ACCEPTED', failed_at=now(), error=repr(error),
                traceback=traceback.format_exc(), manifest_sha256=manifest_hash, runner_sha256=sha(SCRIPT)))
            raise

def self_test():
    rows = np.array([9, 2, 7, 3, 1], dtype=np.int64)
    probability = np.array([.5, .9, .5, .9, .5], dtype=np.float64)
    order, rank = ranked_indices(rows, probability)
    np.testing.assert_array_equal(rows[order][rank], np.array([2, 3, 1, 7, 9]))
    require(int(math.ceil(.005 * 201)) == 2, 'Budget rounding changed')
    # The double values differ below float32 precision; they must not become ties.
    values = np.array([.5000000001, .5000000002], dtype=np.float64)
    order, rank = ranked_indices(np.array([0, 1]), values)
    np.testing.assert_array_equal(rank, np.array([1, 0]))
    # Unequal fold sizes expose the difference between seed-pooled counts and
    # averaging fold precision. The seed result must use the already chosen sets.
    fixture = []
    for seed in range(5):
        for fold in range(5):
            for arm, tp in (('base16', int(fold < 2)), ('base_g32', int(fold < 4))):
                for budget in BUDGETS:
                    n = 201 + fold * 100; k = int(math.ceil(n * budget)); positive = 10 + fold
                    fixture.append(dict(dataset='hi_small', model='mlp', feature_set=arm, seed=seed, fold=fold,
                        budget=budget, k=k, n_test=n, n_positive=positive, tp=tp, precision_at_k=tp / k, recall_at_k=tp / positive))
    frame = pd.DataFrame(fixture, columns=FOLD_COLUMNS); seeds = make_seeds(frame)
    require(len(make_summary(frame)) == 6 and len(seeds) == 30, 'Fold/seed closure failed')
    first = seeds[(seeds.feature_set == 'base16') & (seeds.seed == 0) & (seeds.budget == .005)].iloc[0]
    require(first.tp == 2 and first.precision_at_k == 2 / sum(math.ceil((201 + fold * 100) * .005) for fold in range(5)), 'Seed aggregation must divide pooled counts')
    pair = dict(dataset='hi_small', pair_id='g_minus_base', left_model='mlp', left_feature_set='base_g32', right_model='mlp', right_feature_set='base16')
    pf, ps, summary = pairs_tables(frame, seeds, [pair])
    require((len(pf), len(ps), len(summary)) == (75, 15, 3), 'Pair fold/seed/summary closure failed')
    require((summary.recall_positive_folds == 10).all() and (summary.recall_positive_seeds == 5).all(), 'Descriptive paired counts failed')
    # A synthetic parquet exercises the real cache commit and source-SHA gate.
    with tempfile.TemporaryDirectory(prefix='unified_budget_toy_') as directory:
        directory = Path(directory).resolve(); ledger = directory / 'accounts.parquet'
        account_rows = np.arange(100); labels = (account_rows % 3 == 0).astype(np.int8)
        pd.DataFrame(dict(row_index=account_rows, account_id=[f'toy{i}' for i in account_rows], label=labels)).to_parquet(ledger)
        split_records = []
        for seed in range(5):
            for fold in range(5):
                file = directory / f'seed{seed}_fold{fold}.npz'; np.savez(file, outer_test=np.arange(fold * 20, (fold + 1) * 20))
                split_records.append(dict(seed=seed, fold=fold, path=str(file), sha256=sha(file)))
        canonical = Canonical({'canonical': {'hi_small': {'accounts': dict(path=str(ledger), sha256=sha(ledger)), 'splits': split_records}}}, directory)
        chosen_rows = np.arange(20)[::-1]; scores = (chosen_rows % 4).astype(np.float64) / 4
        prediction = directory / 'toy.parquet'
        pd.DataFrame(dict(row_index=chosen_rows, account_id=[f'toy{i}' for i in chosen_rows], y_true=labels[chosen_rows], y_score=scores)).to_parquet(prediction)
        entry = dict(dataset='hi_small', model='mlp', feature_set='base16', seed=0, fold=0,
            path=str(prediction), sha256=sha(prediction), expected_ap=float(average_precision_score(labels[chosen_rows], scores)))
        result = process_entry(entry, directory, directory / 'output', 'synthetic', canonical)
        require(result == process_entry(entry, directory, directory / 'output', 'synthetic', canonical), 'Bound cache resume changed result')
        original = prediction.read_bytes(); prediction.write_bytes(original + b'changed')
        rejected = False
        try: process_entry(entry, directory, directory / 'output', 'synthetic', canonical)
        except ValueError: rejected = True
        require(rejected, 'Resumed cache must recheck source file SHA before reuse')
    print(json.dumps(dict(passed=True, synthetic_only=True, row_index_tie_rule=True, stored_float64_precision_preserved=True,
        ceil_budget=True, fold_seed_and_paired_aggregation=True, cached_source_sha_rechecked=True,
        formal_predictions_read=False, models_fit=False)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test: return self_test()
    if args.manifest is None or args.output is None: parser.error('--manifest and --output are required')
    return run(args)
if __name__ == '__main__': sys.exit(main())
