# Manuscript analysis map

Table numbers match the 2026-11-07 manuscript with 22 tables. Files below are
relative to `Altman_experiments/`; `R/` denotes `results/`, `C/` denotes `code/`.
Full source filenames, original hashes and any path adaptations are in
`provenance/source_inventory.json`. Current table fragments are under
`paper/tables/` and their labels/captions in `paper/table_inventory.json`.

| Paper item | Research implementation | Accepted/recorded results |
|---|---|---|
| Tables 1–2: dataset/graph structure | C/ibm_pipeline.py; eswa_graph_label_stats_20261012.py | data/dataset_statistics; R/paper_eswa_supplementary_analysis_20261012/graph_label_stats.json |
| Table 3: four-input tree AP | C/run_account_gmkep_tuning.py; run_account_eswa_supplement_trees_20261013.py; run_account_eswa_remote_completion_20261020.py | R/account_gmkep56_tuned_20261008; R/account_eswa_supplement_20261013/validation/tree/complete_settings; R/account_eswa_remote_completion_20261020/final/complete_settings |
| Table 4: paired structural/PH increments | C/run_account_gmkep_tuning.py; report_account_gmkep_multiclassifier.py | R/account_gmkep56_tuned_20261008/paired_feature_summary.csv and fold/OOF values |
| Table 5A: module additions/deletions | C/prepare_eswa_supplement_design_20261013.py; run_account_eswa_supplement_trees_20261013.py; run_account_eswa_remote_completion_20261020.py | Three accepted supplement/completion batches, including R/account_eswa_remaining_20261029/final/complete_settings |
| Table 5B: H0/N/Q controls and conditional H1 | C/build_account_eswa_q10_20261013.py; the same tree runners | Same three batches: verified_paired_summary.csv, lhs struct_n46/struct_q10_54 minus gmke44 for the new N/Q rows |
| Table 6: Full-model SHAP | C/eswa_supplementary_analysis_20261012.py, shap part | R/paper_eswa_supplementary_analysis_20261012/shap_*_fold_values.csv and summaries; LightGBM/XGBoost only |
| Table 7: 1% budget | C/analyze_unified_alert_budgets_20261028.py; tools/audit_results.py | R/account_unified_alert_budget_20261028/final plus R/account_gfp_alert_budget_extension_20261106/combined_budget_summary.csv |
| Tables 8, C.5: typology recall | C/eswa_supplementary_analysis_20261012.py, patterns part | R/paper_eswa_supplementary_analysis_20261012/pattern_seed_values.csv and pattern_summary.csv; three inputs/three tree models |
| Table 9: fixed-parameter paired increments | C/run_account_gmkep_multiclassifier.py | R/account_gmkep56_multiclassifier_20261008 |
| Table 10A: Full/GFP under three classifiers | C/build_account_eswa_gfp_20261013.py; build_account_eswa_gfp_float32_cache_20261013.py; tree runners | XGBoost GFP: R/account_eswa_remote_completion_20261020/final/complete_settings; LightGBM/RF GFP: R/account_eswa_remaining_20261029/final/complete_settings |
| Table 10B: MLP four inputs | C/run_account_mlp_cuda_paired_20261022.py | R/account_mlp_cuda_paired_20261022/finalization_20261023/validation/complete_settings |
| Table 10C: GraphSAGE four inputs | C/run_account_graphsage_cuda_optimized_20261018.py; run_account_graphsage_cuda_features_20261018.py; checkpoint helpers | R/account_graphsage_cuda_optimized_20261018/final_results_20261020/validation/complete_settings |
| Tables A.1–A.3: feature dictionary | C/ibm_pipeline.py; build_account_s10_features.py; account_s10_math.py | Main protocol feature_columns; supplement feature_registry.json; table fragments |
| Table A.4: construction complexity | C/eswa_graph_complexity_stats_20261012.py | R/paper_eswa_supplementary_analysis_20261012/graph_complexity_stats.json |
| Table B.1: workflow/tuning decomposition | C/run_account_gmkep_multiclassifier.py; run_account_gmkep_tuning.py | Fixed and tuned fold/paired tables; explicitly limited to the 18 original 16/44/56 settings |
| Table C.1: threshold metrics, four inputs | C/run_account_gmkep_tuning.py; supplement/completion tree runners; tools/audit_results.py | Main 18 settings plus six base_g32 settings; all 24 current rows included |
| Table C.2: fixed Base+G vs Struct | C/run_account_graph32_completion_20261009.py | R/account_graph32_completion_20261009/fixed_summary |
| Tables C.3–C.4: other budgets/paired recall | C/analyze_unified_alert_budgets_20261028.py; summarize_unified_alert_budgets_20261028.py | R/account_unified_alert_budget_20261028/final; GFP extension completes all 46 settings for C.3 |
| Table C.6: typology structural profiles | C/eswa_graph_label_stats_20261012.py | R/paper_eswa_supplementary_analysis_20261012/graph_label_stats.json |
| Table C.7: fitting/wall/inference times | Tree/neural fit timers; C/eswa_supplementary_analysis_20261012.py, timing part | results/timing: tree, mlp, graphsage, gfp_xgboost and gfp_lightgbm_random_forest; all three GFP models included |
| Figure 1: literature positioning | paper/figure_sources/milestones_figure_20261029 | paper/figures/fig01_research_milestones.* |
| Figure 2: descriptor framework | paper/figure_sources/topology_features_20261103 | paper/figures/fig02_topoaml_framework.* |
| Figure 3: fixed/tuned paired AP | Inline TikZ uses Tables 4/9 values | paper/figures/fig03_paired_ap.tex |

The portable audit reconstructs reported 46-setting model and 138-row budget
summaries from included fold-level results. The full paired, OOF, SHAP, typology
and timing source tables are also retained for inspection. Dataset-level graph
statistics and feature dictionaries are not model-comparison results and should
not be expanded with GFP rows.
