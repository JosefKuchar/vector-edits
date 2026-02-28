# VectorEdits: A Dataset and Benchmark for Instruction-Based Editing of Vector Graphics

This repository contains research code for generating the VectorEdits dataset and evaluating models on the VectorEdits benchmark.

- [Hugging Face Dataset](https://huggingface.co/datasets/mikronai/VectorEdits)
- [Paper](https://arxiv.org/abs/2506.15903)

## Repository Scripts

- `cluster.py` - finds visually similar SVG pairs inside collections (using CLIP embeddings) and saves candidate edit pairs.
- `labeling.py` - generates natural-language editing instructions for SVG pairs using a vision-language model through OpenRouter.
- `evaluate_part_1.py` - runs an editing model on benchmark instructions and stores the produced edited SVG outputs.
- `evaluate_part_2.py` - computes evaluation metrics (CLIP similarity, DINOv2 similarity, MSE, invalid SVG count) for generated SVG edits.
- `dataset.ipynb` - exploratory notebook used during dataset work and analysis.

## Environment Notes

- Scripts assume GPU execution (`CUDA_VISIBLE_DEVICES` is set inside scripts).
- Some scripts expect datasets to exist at local paths under `/var/tmp/xkuchar/...`.
- API-based scripts require `OPENROUTER_API_KEY` in your environment (for example via `.env`).

> The source code in this repository is intended for research and experimentation. It is **not production-ready**.
