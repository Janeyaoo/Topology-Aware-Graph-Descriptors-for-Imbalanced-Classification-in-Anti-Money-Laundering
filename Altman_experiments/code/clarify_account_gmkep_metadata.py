"""Record unused upstream catalog blocks without editing a frozen protocol."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "Altman_experiments/results/account_gmkep56_multiclassifier_20261008"
BLOCKS = ["base16", "G16", "M5", "K3", "E4", "P12u"]


def clarify(out):
    path = out / "protocol.json"
    protocol = json.loads(path.read_text())
    record = {
        "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "Module catalog clarification only; frozen protocol, model inputs, splits, parameters and numerical data unchanged.",
        "authoritative_input_field": "protocol.json/feature_columns",
        "actual_blocks": {key: values for key, values in protocol["feature_blocks"].items() if key in BLOCKS},
        "unused_upstream_catalog_blocks": ["S_graph4", "S_hodge4", "S_PL2"],
        "actual_feature_counts": protocol["feature_sets"],
        "contains_Q_or_S_in_actual_inputs": any(name.startswith(("q_", "s_")) for columns in protocol["feature_columns"].values() for name in columns),
    }
    assert list(record["actual_blocks"]) == BLOCKS
    assert [name for columns in record["actual_blocks"].values() for name in columns] == protocol["feature_columns"]["gmkep56"]
    assert record["actual_feature_counts"] == {"base16": 16, "gmke44": 44, "gmkep56": 56}
    assert record["contains_Q_or_S_in_actual_inputs"] is False
    destination = out / "metadata_clarification.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == record, "Existing clarification differs; do not overwrite"
        status = "existing_verified"
    else:
        destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        status = "created"
    print(json.dumps({"status": status, "clarification": str(destination.relative_to(ROOT))}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT)
    clarify(out)


if __name__ == "__main__": main()
