# Data

These five CSV files are included in the repository. Generated predictions, arrays, checkpoints, and other experiment outputs are excluded by `.gitignore`.

| File | Rows | Columns |
| --- | ---: | --- |
| `combined_emotions.csv` | 9,870 | `emotion`, `text`, `dataset`, `input_text` |
| `train_df.csv` | 2,400 | `input_text`, `emotion` |
| `val_df.csv` | 1,974 | `input_text`, `emotion` |
| `emoevent_en.csv` | 7,303 | `id`, `tweet`, `emotion`, `offensive`; **tab-separated** |
| `xed-en.csv` | 2,803 | `text`, `emotion` |

`emotion` is the gold label. In the combined dataset, `text` is the original sentence, `dataset` identifies its source, and `input_text` contains the sentence in the original emotion-classification prompt. The combined sources are ISEAR (5,395), enISEAR (715), enVent (3,300), and EXPRESS (460). The upstream EXPRESS filtering code is outside this release.

The bundled model-prediction columns and redundant model-specific dataset copy have been removed. The combined dataset retains every source row and its original order. Running activation extraction creates fresh per-model `combined_zeroshot_preds_<model>.csv` files with `pred_emotion`, plus correctly predicted subsets and activation arrays for subsequent analysis. Preserve generated row indices when indexing those arrays.

The shared split files are unchanged: 2,400 training rows (400 per emotion) and 1,974 held-out rows. They match the stratified 80/20 split with seed 42, followed by training-class sampling and shuffling. Do not split the 2,400-row training file again. `input_text` already contains a prompt; steering uses it directly, while prompt robustness matches back to the combined dataset's raw `text`.

The original split has two prompt texts shared across training and held-out rows. The training code uses `val_df` for checkpoint selection. The lexical control is constructed from the full combined dataset and includes training examples; it is a full-corpus diagnostic. These behaviors are preserved.

External evaluation uses only EmoEvent and XED. Read EmoEvent with `sep="\t"`; its 3,305 `others` rows are excluded, leaving 3,998 evaluation examples.
