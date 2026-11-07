# Reproduction levels and research workflow

## 1. Saved-result audit — works from this folder

```bash
python3 tools/audit_results.py --out reproduced_results
```

This has no external Python dependencies. `MANIFEST.json` checks the packaged
bytes, while `provenance/source_inventory.json` records original source hashes.
The generated model and budget CSVs can be compared to the accepted summaries.
The tool refuses an existing output directory.

## 2. Feature/math inspection

The preserved research code is under `Altman_experiments/code`, keeping the
original relative project layout. It contains the actual capped BFS geometry,
GF(2) PH calculations, structural features, nested tuning, GFP adaptation,
CUDA MLP/GraphSAGE implementations and independent verifiers.

With NumPy installed, a small graph-math verification can be run without data:

```bash
python Altman_experiments/code/verify_account_s10_math.py
```

The source builder includes development spectral extensions beyond the
manuscript. The manuscript uses only the 16 transaction, 28 structural and
12 PH columns named in the main protocol and feature dictionary, not every
column that the development cache can produce.

The independent feature verifier reads four pure functions from the historical
`02_build_features_vr_persistence.py` oracle with AST extraction; that source is
included and its top-level historical pipeline must not be imported directly.

## 3. Complete feature/model regeneration — needs omitted inputs

Read the corresponding protocol and command's `--help` before execution. Do not
run trainers against the accepted folders shipped here. Use a separate work
copy and new output directory; the scripts intentionally reject overwritten or
hash-inconsistent registrations.

The original staged workflow is:

1. Obtain HI-Small and LI-Small transactions and pattern annotations. The
   preprocessing/transaction and initial structural-feature implementation is
   `ibm_pipeline.py --variant HI-Small` or `--variant LI-Small`. That historical
   script also runs older classification stages; only its normalized transaction
   and structural caches are prerequisites of this manuscript's later stage.
2. Construct the shared cap-40 geometry and PH cache using
   `build_account_s10_features.py --phase prepare`, followed by `--phase build
   --dataset hi_small` and `--dataset li_small`. Input caches must occupy the
   documented relative paths. The builder records exact input hashes, account
   order, feature names and graph identities.
3. Build the fixed and tuned 16/44/56 tree stages with
   `run_account_gmkep_multiclassifier.py` and `run_account_gmkep_tuning.py`.
   Their `prepare` stages register candidates, splits and source hashes before
   fitting. Each fit uses the recorded class weights and seed formulas.
4. Build the extra geometry counts with `build_account_eswa_q10_20261013.py`,
   register supplement arms with `prepare_eswa_supplement_design_20261013.py`,
   and run the supplement tree runner. `run_account_graph32_completion_20261009.py`
   handles the recorded fixed Base+G comparison; the supplement runner handles
   independently tuned Base+G and controls.
5. For GFP, obtain Snap ML 1.17.2 for a supported platform and provide a native
   engine/wheel verification record required by
   `build_account_eswa_gfp_20261013.py`. This produces the stream descriptors;
   the float32 cache helper prepares the tree input. The package does not
   redistribute the wheel or bypass the native-validation gates.
6. MLP and GraphSAGE require the complete graph/input package: canonical account
   order, raw feature arrays, labels, outer/inner splits, CSR graph and source
   bindings. The CUDA runners accept explicit input/output paths. They are not
   stand-alone learners that can be launched using metric CSVs alone.
7. Validate each trained setting with its independent verifier before
   aggregating. After scores/models are restored or regenerated, use the
   supplementary analysis code for SHAP/typology/timing and the unified budget
   code for saved-score budgets. No partial-setting mean should be reported.

Because the large registered caches/scores/models are omitted, this release does
not provide a single command that can reproduce all training directly from the
included files. Do not interpret an archived registration's hashes as a promise
that redacted transport metadata or locally rebuilt artifacts have the same
bytes. A new execution must register its own inputs and source code and be
validated afresh. The current release was checked through saved-result
reaggregation and source parsing, not a costly end-to-end training rerun.

## Figure and table sources

The two schematic figure PDFs/SVGs/PNGs are final manuscript assets. Their
source generators are supplied under `paper/figure_sources`; output paths were
historically tied to the original figure-building layout, so set them to a new
build location before regeneration. Font availability and TeX tool versions can
change exact rendering. No font files are redistributed.

Tables and Figure 3 are TeX fragments, not standalone documents. A compiling
document needs booktabs, threeparttable, multirow, xcolor/colortbl, array,
graphicx and PGFPlots/TikZ, plus the original macros `controlAP`, `totalSHAP` and
`incrementAP`. The snippets preserve the final scientific content; they do not
include an author-identifying full manuscript. `paper/macros.tex` supplies the
packages, colors and three table macros for embedding the fragments.

## Portability edits

Three analysis scripts had a personal absolute root replaced by
`Path(__file__).resolve().parents[2]`. A small amount of MLP transport metadata
and original timing-path provenance was normalized to relative/public paths.
The framework figure generator uses its fixed fallback style directly instead
of executing a workstation-local style helper.
The source inventory marks these files and preserves both hashes. No model
parameter, seed, metric or scientific result was changed by those edits.
Four preparation functions also skip absent documentation-only snapshots of the
old project-wide reports (`EXPERIMENT_RESULTS_SUMMARY*` and the old framework
draft). The report renderer/template source is retained for provenance, but its
legacy cross-project report generation is not a supported reproduction entry
point here. This avoids fabricating placeholder reports or bundling unrelated
experiments. No scientific-input or validation check was removed.
Scheduling services, SSH helpers, credentials, connection monitors and machine
recovery utilities are intentionally excluded.
