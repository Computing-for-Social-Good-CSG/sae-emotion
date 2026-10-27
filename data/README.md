# Data

The repository includes the following source CSVs:

| File | Rows | Columns |
| --- | ---: | --- |
| `combined_emotions.csv` | 9,870 | `emotion`, `text`, `dataset`, `input_text` |
| `train_df.csv` | 1,919 | `input_text`, `emotion` |
| `val_df.csv` | 479 | `input_text`, `emotion` |
| `test_df.csv` | 1,974 | `input_text`, `emotion` |
| `feature_discovery_train.csv` | 1,917 | `emotion`, `text`, `dataset`, `input_text` |
| `emoevent_en.csv` | 7,303 | `id`, `tweet`, `emotion`, `offensive`; **tab-separated** |
| `xed-en.csv` | 2,803 | `text`, `emotion` |

## Combined dataset

`emotion` is the gold label, `text` is the original sentence, and `dataset` identifies its source. `input_text` contains the sentence in the original emotion-classification prompt.

The combined sources are ISEAR (5,395), enISEAR (715), enVent (3,300), and EXPRESS (460). For steering feature discovery, extraction starts from `feature_discovery_train.csv`; the combined file remains the source of raw text and dataset metadata.

Extraction generates per-model `combined_zeroshot_preds_<model>.csv` files containing `pred_emotion`, correctly predicted subsets, and activation arrays. Keep the generated row indices when indexing those arrays. Generated predictions, arrays, checkpoints, and other experiment outputs are excluded from Git by `.gitignore`.

## Steering splits

The original stratified 80/20 partition (seed 42) supplies the unchanged 1,974-row `test_df.csv`. From the original balanced 2,400-example training pool, two examples with sentences also present in the test set are excluded. Unique prompt groups are split 80/20 with seed 42, stratifying by each group's first emotion label. All copies of a sentence stay together, including copies with different labels. This yields 1,919 training rows and 479 validation rows, with no sentence overlap across splits.

`val_df.csv` is used only for checkpoint, layer, step-budget and multiplier selection. Final main-dataset scores use `test_df.csv`. `feature_discovery_train.csv` contains the training sentences with raw text and source metadata, deduplicated by prompt and label; it excludes validation and test sentences.

Steering consumes the existing prompted `input_text`. Prompt robustness matches test prompts and labels to the combined dataset's original `text` before applying an alternative template. Lexical-control preprocessing still uses the full combined dataset and is a separate diagnostic, not an independent test set.

## External datasets

External evaluation uses EmoEvent and XED. Read EmoEvent with `sep="\t"`; its 3,305 `others` rows are excluded, leaving 3,998 evaluation examples. XED contains 2,803 evaluation examples.
