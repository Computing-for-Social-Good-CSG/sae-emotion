# Research history

These files preserve distinct source variants without presenting them as canonical final-paper implementations.

- `legacy/`: earlier targeted steering, coordinate-search emotion vectors, LoRA (not a reported final-paper baseline), steering with few-shot prompts, and an incomplete clean-path-patching driver.
- `prompt_discovery/`: prompt-specific feature re-discovery experiments and their local helper versions. Some filenames/directories disagree with the prompt chosen inside the code. Keep these separate from `notebooks/evaluation/prompt_robustness.ipynb`, which evaluates existing steering checkpoints across prompts.
- `checkpoints/`: unique code or notebook cell sources recovered from nested notebook-backup ZIPs. Exact source duplicates and empty checkpoint files were not expanded into additional files. A source-duplicate entry in the migration manifest points to its retained counterpart; notebook outputs may have differed.
- `lexical_control/`: original and earlier resume evaluators. The main workflow uses the later variant with resume support and cache handling; no scientific performance ranking is implied.

All retained notebooks have cleared outputs and removed credentials. The original ZIP preserves historical outputs if needed. Cluster job examples retain their resource requests but use `python` from your activated environment; their partition/GPU values are examples from the original machine, not portable settings.

The missing module `clean_path_patching_utils.py` prevents the older clean-path runner from executing. No replacement implementation has been fabricated.
