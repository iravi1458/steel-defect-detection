# Bias-Aware Hybrid CNN–Transformer Ensemble for Multilingual Offline Signature Verification

Writer-independent offline signature verification across Hindi, Bengali, and English scripts, using a calibrated score-level ensemble of a convolutional (ResNet-18) and a convolution–Transformer (CvT-13) backbone, evaluated under a leakage-controlled writer-disjoint protocol.

M.Sc. thesis, Department of Computer Science, Bishop's University (2026). Supervisor: Dr. Mohammed Ayoub Alaoui Mhamdi.

---

## Problem

Offline signature verification is used in banking, legal, and identity-management settings, but two problems recur in the literature:

1. **Writer leakage.** Many reported results come from splits where the same writer appears in both training and test data, so the metric measures writer memorisation as much as verification ability.
2. **Heuristic fusion.** When multiple backbones are combined, fusion weights are usually picked by a fixed rule, without first establishing that the backbones are actually complementary or checking how the fused system behaves across scripts.

This project addresses both, and — importantly — reports what the evidence supports rather than assuming the proposed method wins.

## Approach

- **ResNet-18 branch** — convolutional feature extraction for local stroke structure
- **CvT-13 branch** — convolution–Transformer branch for long-range spatial dependencies
- **Siamese pairwise verifier** — each branch scores a reference–query pair via `[|z_r − z_q| ‖ z_r ⊙ z_q]` into a verification head
- **Platt calibration** — each branch's logits calibrated on validation only, placing both in a common log-odds space (monotonic, so individual EER/AUC unchanged)
- **BAES fusion** — Bias-Aware Ensemble Selection picks the fusion weight by minimising `J(ω) = EER_val(ω) + α · std_c(EER_val^(c))`, generalising ordinary weighted fusion (recovered at α = 0)
- **SSWDS protocol** — Strict Script-aware Writer-Disjoint Splitter partitions *writers*, not samples, with per-script proportions preserved and a fixed seed for reproducibility

## Datasets

| Corpus | Script | Writers | Per writer |
|---|---|---|---|
| BHSig260 | Hindi | 160 | 24 genuine, 30 skilled forgeries |
| BHSig260 | Bengali | 100 | 24 genuine, 30 skilled forgeries |
| CEDAR | English | 55 | 24 genuine, 24 skilled forgeries |

Datasets are not redistributed here. See the original sources (Indian Statistical Institute for BHSig260; CEDAR, University at Buffalo).

---

## Results

All results on 1,962 writer-disjoint test trials (Hindi 1,008 / Bengali 630 / English 324), thresholds selected on validation and applied unchanged at test.

### RQ1 — Does the evaluation protocol matter?

| Protocol | Train∩Val | Train∩Test | Val∩Test | AUC (seed 42) | EER (seed 42) | AUC (5-seed) | EER (5-seed) |
|---|---|---|---|---|---|---|---|
| SSWDS | 0 | 0 | 0 | 0.870 | 0.193 | 0.834 | 0.228 |
| Writer-random | 308 | 307 | 307 | 0.838 | 0.229 | 0.835 | 0.230 |

The writer-random split leaked over 300 writers across every pair of partitions. On a single seed SSWDS looked clearly better — but that gap collapsed once averaged over five seeds. The real difference is **trustworthiness of measurement, not achievable performance**: writer-random scores mix verification ability with writer memorisation, and are less stable across seeds (test AUC std 0.0219 vs 0.0169).

### RQ2 — Are the branches complementary?

| Branch | EER | AUC | Hindi EER | Bengali EER | English EER |
|---|---|---|---|---|---|
| ResNet-18 + contrastive | 0.0404 | 0.9944 | 0.0667 | 0.0244 | 0.0 |
| CvT-13 + contrastive | 0.0808 | 0.9792 | 0.0931 | 0.0489 | 0.0 |

Complementarity, measured directly rather than assumed:

| Scope | Disagreement | Double-fault | Yule Q | N10 | N01 |
|---|---|---|---|---|---|
| Overall | 0.0872 | 0.0148 | 0.806 | 128 | 43 |
| Hindi | 0.1250 | 0.0248 | 0.741 | 91 | 35 |
| Bengali | 0.0714 | 0.0063 | 0.774 | 37 | 8 |
| English | 0.0000 | 0.0000 | — | 0 | 0 |

Yule Q = 0.806 indicates the branches are only **modestly** complementary — they largely succeed and fail on the same trials. Of 171 disagreements, 128 are simply ResNet-18 being right where CvT-13 is wrong, consistent with ResNet-18 being the stronger branch rather than the two capturing distinct failure modes. This predicts a modest fusion gain, which is what RQ3 finds.

### RQ3 — Does bias-aware fusion help?

| Configuration | EER | AUC | FAR | FRR | ACC | Cross-script std |
|---|---|---|---|---|---|---|
| ResNet-18 + contrastive | 0.0404 | 0.9944 | 0.0274 | 0.0590 | 0.9633 | 0.0275 |
| CvT-13 + contrastive | 0.0808 | 0.9792 | 0.0794 | 0.0816 | 0.9200 | 0.0380 |
| Simple average | 0.0382 | 0.9949 | 0.0346 | 0.0365 | 0.9648 | **0.0250** |
| Weighted fusion (w=0.65) | **0.0368** | **0.9954** | 0.0325 | 0.0451 | 0.9638 | 0.0267 |
| BAES (α=0.25, w=0.65) | **0.0368** | **0.9954** | 0.0325 | 0.0451 | 0.9638 | 0.0267 |

Fusion improves on the stronger individual branch: **8.9% relative EER reduction** over ResNet-18 alone.

**On BAES specifically — a null result, reported as such.** The weight search returns w\* = 0.65 for every α ≤ 7, including the reported α = 0.25. At these settings BAES selects *exactly the same weight* as ordinary validation-EER-minimising fusion, which is why the two rows above are identical. The disparity term is not inert — the minimiser shifts to w\* = 0.55 at α = 8, confirming BAES is a genuine generalisation rather than a degenerate one — but the trade-off it exposes is unfavourable on this corpus: that shift reduces cross-script standard deviation only from 0.0172 to 0.0169 while degrading validation EER from 0.0389 to 0.0419, roughly a tenfold larger movement in accuracy than in balance. The accuracy-optimal weight was already near disparity-optimal for this branch pair. The simple-average baseline, with no balancing mechanism at all, happens to achieve the lowest cross-script spread.

This is reported as an honest empirical finding rather than reframed as a success. Whether BAES delivers a favourable trade-off on a corpus with genuine cross-script imbalance remains untested.

---

## Repository structure

```
resnet18_pipeline.ipynb   ResNet-18 branch: training, calibration, evaluation
cvt13_pipeline.ipynb      CvT-13 branch: training, calibration, evaluation
ensemble_baes.ipynb       Score-level fusion, BAES weight search, per-script analysis
signature_common.py       Shared preprocessing, SSWDS splitter, trial construction, metrics
complementarity.py        Disagreement, double-fault, and Yule Q statistics
experiments/              Earlier experimental notebooks (baselines, ablations, Swin, RF/XGBoost variants)
```

## Stack

Python · PyTorch · timm · scikit-learn · NumPy · pandas · OpenCV · Matplotlib

## Limitations

- Three scripts from two corpora; the English test subset (324 trials) is smaller than Hindi (1,008) or Bengali (630), which limits precision of the per-script disparity estimate for English.
- Only two backbones evaluated — the observed complementarity may not generalise to other CNN/Transformer pairings.
- Score-level fusion only; feature-level, decision-level, and attention-based fusion are future work.

## Citation

```bibtex
@mastersthesis{ranjan2026baes,
  title  = {Bias-Aware Hybrid CNN--Transformer Ensemble for Multilingual,
            Writer-Disjoint Offline Signature Verification},
  author = {Ranjan, Ravi},
  school = {Bishop's University},
  year   = {2026}
}
```

## License

Thesis released under CC BY-SA 4.0.

## Author

**Ravi Ranjan** — M.Sc. Computer Science, Bishop's University
