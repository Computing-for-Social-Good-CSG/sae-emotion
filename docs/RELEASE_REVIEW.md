# Release review and author questions

This review compares the supplied ZIP and additional uploads with the supplied camera-ready PDF. Findings concern these particular files; the final implementations may exist elsewhere. The organizing pass preserves research logic except for the requested dataset scope, use of the supplied shared splits, and the documented correction to nested prompts in robustness evaluation. These changes and remaining differences are recorded below.

## Agreed release scope

- **Generate large arrays locally.** `.npy` files are intentionally omitted. Readers run activation extraction and the subsequent selection/training stages themselves. Existing server files need not be uploaded or moved.
- **Generate checkpoints/results from source.** The release does not require uploaded pretrained steering artifacts. Small feature lists, run configurations, and metrics would still be useful optional reference outputs, but absent artifacts are not filled with inferred or fabricated results.
- **EXPRESS preprocessing excluded for now.** The supplied combined dataset is the starting point. The absence of the upstream EXPRESS filtering code is a scope decision, not a request for more files.
- **Retain steering variants.** Keep the 400-step and 600-step emotion-vector notebooks and other distinct research variants for readers to run. No version is declared best from unmatched saved test scores. If a default is later selected, use a common validation protocol and record the choice before test evaluation.
- **Two external datasets only.** Active global-steering defaults and RQ3 analysis now use XED and EmoEvent. Emotion-cause evaluation is outside scope; older archived code is historical and is not an active entry point.
- **Shared steering split.** The author confirmed that all steering uses the same train/test split. Active learned steering uses the supplied training/held-out files, and the targeted multiplier sweep now evaluates the supplied held-out file instead of the full corpus.

## Resolved by the additional uploads

- **Lexical-shortcut and balancing code is now present.** The new preprocessing and checkpoint-evaluation notebooks are integrated under `notebooks/evaluation/`. CPU execution on the supplied main CSV reproduced 1,332 balanced rows (222 per emotion). See [LEXICAL_CONTROL.md](LEXICAL_CONTROL.md) and `results/validation/lexical_preprocessing.json`.
- The main lexical evaluator is the version with resume support and cache handling; older evaluation variants remain in `archive/lexical_control/`. This is an operational choice, not a model-performance ranking.
- The steering utility re-upload is byte-identical to the original version; the robustness notebook's cell source is also unchanged. Duplicate implementations were not added.
- **Cluster package versions are recorded.** The two environment exports agree on all 245 packages. Python is 3.11.14; PyTorch is 2.13.0+cu126; SAELens is 6.37.3. Direct-dependency pins and the original exports are included. GPU model/driver were unavailable in this capture; the author subsequently confirmed NVIDIA H200, recorded separately in `environment/hardware.json`.
- **The saved splits are supplied.** `train_df.csv` contains 2,400 rows (400 per emotion); `val_df.csv` contains 1,974 rows. Both exactly match the original stratified 80/20 split, seed 42, and training balancing/shuffle, including row order. They are copied byte-for-byte. Global steering now loads them directly, both emotion-vector notebooks already reference these paths, and prompt robustness verifies every run's saved holdout against the canonical upload. See `results/validation/split_consistency.json`.
- **Nested prompts are corrected in the active robustness notebook.** After checking each run's saved split, the evaluator matches each held-out `input_text`/`emotion` pair to the main CSV's original `text` and applies each alternative prompt once. All 1,974 rows, labels, order, and duplicates are preserved. Missing or ambiguous mappings fail explicitly. CPU checks verified all 5,922 alternative prompts; see `results/validation/prompt_robustness_inputs.json`. Corrected scores still require a GPU rerun and use `data/prompt_robustness_evaluation_raw_text/`, including table exports.

## Remaining information for exact reproduction

1. **GPU/run provenance.** Package versions and the author-reported NVIDIA H200 GPU are now recorded; the NVIDIA driver version remains unspecified. The report captures the current environment, so any changes since the final paper runs should be noted. See [ENVIRONMENT.md](ENVIRONMENT.md).
2. **Final protocol and results.** Confirm the model/SAE settings, learning rates, feature-selection rule, checkpoint selection, and statistical-testing scope identified below. Shared split membership is resolved; its role in checkpoint selection versus final reporting still needs clarification. Prompt construction is corrected, but results need rerunning before they can be attributed to the corrected code. Readers can rerun source, but these choices determine whether they reproduce the published experiment.
3. **Data provenance and license.** Confirm dataset source versions and redistribution terms and choose the code license. Supplied CSVs remain available locally and ignored by Git. Upstream EXPRESS filtering code is outside current scope.

## Code/paper differences to resolve

| Topic | Evidence in the supplied material | Question / impact |
| --- | --- | --- |
| Feature discovery population | Section 4 describes means over emotion-labeled instances; `select_candidates.py` reads `combined_zeroshot_filtered_*.csv`, and the extractor groups `pred_emotion` on correctly predicted rows. | Was correct-only feature discovery used in the final experiments? If so, make the paper and release instructions explicit. |
| Entropy grouping | Paper describes maximum-posterior GMM assignments. `estimate_entropy_threshold_gmm` instead minimizes the absolute difference of component densities on a grid over [0,1], then labels values below that threshold. | Confirm the final function. A minimum density difference in a near-zero tail need not be the intended component boundary. No scientific correction was silently applied. |
| Joint-ablation grid / plateau | Paper grid ends with 30. Multiple notebooks/scripts use 20; the helper chooses the smallest k reaching 80% of the maximum valid signed drop, with a minimum signed drop of 0.1. | Which exact grid and selection rule generated the published feature lists? |
| Held-out selection | Appendix A.3 describes an 80/20 train/test split. The supplied files and shared use are now confirmed. The steering utility evaluates `val_df` during training and saves the checkpoint with best `main_val` macro-F1. | Is there a separate untouched test set, or did reported scores use the checkpoint-selection split? This question concerns selection/reporting, not whether models share the split. Existing selection behavior is retained. |
| Duplicate texts across splits | The supplied training and held-out files share two distinct exact prompt texts; one text-label pair matches. Training/held-out files have respectively four/five duplicate text-label rows beyond the first. | These are original row-based splits, not fully text-disjoint splits. No deduplication or reassignment was performed; document the duplicates, or evaluate a new deduplicated protocol separately if needed. |
| Lexical-control evaluation population | Preprocessing uses the full 9,870-row combined CSV. The reproduced 1,332-row balanced control contains 403 distinct training prompt texts (404 rows) and 275 distinct held-out prompt texts (275 rows). | This control is not an independent held-out set. Describe it as a filtered full-corpus diagnostic, or construct and rerun a separate held-out control. Filtering behavior was preserved. |
| Prompt-robustness result provenance | The original supplied evaluator nested alternative templates around already-prompted main inputs. The active notebook now recovers raw sentences before applying an alternative template; CPU input checks pass. | Rerun GPU evaluation and inspect the new results before updating the paper. No existing score has been relabeled as a corrected result; which code produced the published numbers remains unverified. |
| Learning-rate naming | The provided global runner sets `LR = 1e-2` but writes to a directory named `lr_5e-3_l1_...`; evaluation notebooks expect that name. | Recover actual final run configs. Directory names alone cannot establish the learning rate. |
| Targeted feature scope | Paper describes amplifying all identified features for a target emotion. The supplied runner's `load_exclusive_feature_groups_for_model` discards features affecting more than one emotion. | Were shared features intentionally excluded? Confirm before harmonizing the description or implementation. |
| Significance correction | Appendix A.4 applies Holm separately within blocks of five comparisons. The analysis notebook's `run_all_tests` applies Holm once across all available metrics/datasets/comparisons. | Supply the final Table A.7 script and full-precision inputs; do not regenerate significance from rounded paper numbers. |
| Model coverage | Several drivers keep Gemma-3-1B menus or only subsets of the seven final models. The mediation driver omits some paper settings. | Confirm final sweep settings; retained defaults are individual historical runs. |
| Prompt-discovery variants | Files inside `prompt4/` sometimes set `PROMPT_NAME = "prompt3"`. | Are these stale exploratory copies? They remain in `archive/`, separate from the paper's checkpoint-based prompt robustness evaluator. |

## Other concrete gaps

- `archive/legacy/clean_path_patching.py` imports `clean_path_patching_utils`, which is absent even from the nested checkpoint archives. This older route is retained as incomplete history; the adjacent-group mediation utility is present.
- External EmoEvent is tab-separated even though its filename ends in `.csv`; its rows also include `others`, which the existing evaluation loaders exclude. Its content was not converted.
- The source ZIP contains embedded Hugging Face tokens. They were removed from all deliverable code/notebooks. Revoke/replace the original tokens; the original ZIP still contains them and should not be published.

## Changes made in this pass

- Created an importable `sae_emotion` package and organized experiment entry points by RQ.
- Removed download-number suffixes from Python modules and dataset filenames; updated active imports.
- Preserved distinct baselines, prompt-discovery copies, and unique checkpoint source under explicit names.
- Removed notebook outputs, execution counts, widget/machine metadata, and embedded credentials; retained cell source/order except documented mechanical edits.
- Replaced cluster cache locations with environment-based user-cache defaults; replaced the cluster-specific Python executable in job examples.
- Normalized SAE activation/feature-folder suffix handling in the active extraction, candidate, and logit entry points.
- Added a lightweight launcher, offline validation, data inventory, reproduction guide, and migration manifest.
- Added the lexical-control workflow, CPU preprocessing entry point, and a successful dataset-construction check; added a target-environment version collector.
- Added the author's shared split files, verified historical split equivalence, prevented resplitting saved training rows, aligned targeted evaluation to the shared holdout, and restricted active external evaluation/analysis to XED and EmoEvent.
- Corrected main-data prompt robustness to recover original sentences by exact prompt/label match before applying one alternative template. Preserved split order/duplicates and separated corrected output paths from older runs.
- Left raw data contents byte-identical and protected local data/caches/checkpoints from accidental Git additions.

## Validation limits

See `validation_report.json` for executed static/data checks, `results/validation/lexical_preprocessing.json` for CPU preprocessing, `results/validation/split_consistency.json` for split-loading and overlap checks, and `results/validation/prompt_robustness_inputs.json` for corrected prompt construction. The filtered balanced dataset count and original seeded split were reproduced. Data-loading and prompt-building functions were exercised on CPU in isolation from GPU imports. No large model was downloaded, no CUDA experiment was run, and no reported model score was independently reproduced. GPU package compatibility, final scientific settings, and full numerical reproducibility remain unverified.
