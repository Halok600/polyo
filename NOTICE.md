# Third-party data notice

This project's code is MIT-licensed (see `LICENSE`). The **trained model
artifacts** shipped or described in this repo are trained in part on
third-party datasets that carry their own terms, which take precedence over
MIT for anything derived from them:

- **[BigO(Bench)](https://github.com/facebookresearch/bigobench)** (Meta,
  [arXiv:2503.15242](https://arxiv.org/abs/2503.15242)) — **CC-BY-NC**.
  Any model weights trained using BigO(Bench) labels are for **non-commercial
  use only**, per that dataset's license. This repo, its deployed demo, and
  its Vercel/Render hosting are all non-commercial, so the whole stack is
  consistent — but if you fork this for commercial use, retrain on the
  synthetic generator (`data/synth.py`) plus permissively-licensed sources
  only, and keep BigO(Bench) as an eval-only benchmark.
- **[CodeComplex](https://aclanthology.org/2025.findings-emnlp.1069/)**
  (EMNLP Findings 2025, [arXiv:2401.08719](https://arxiv.org/abs/2401.08719))
  — used for training and as an external comparison point; see the paper for
  its license terms.

This file is expanded as data sources are added in Phase 2/4 ingestion.
