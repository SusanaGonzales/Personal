# Homework 3 — code

Coding for Section 4 of Homework 3 (15 points): matrix factorization on synthetic panel data, plus self-attention from scratch.

## Files

| File | Purpose |
| ---- | ------- |
| `homework_3_solution.ipynb` | **Instructor / answer key** — fully worked solution |
| `homework_3_student.ipynb`  | **Student version** — same structure with `TODO` scaffolds |
| `_build_notebook.py`        | Generator script. Edit cells here and re-run to regenerate **both** notebooks |

The two `.ipynb` files are **always generated from `_build_notebook.py`** — the script is the single source of truth. Do not edit the notebooks directly; your edits will be overwritten on the next regeneration.

## Dependencies

```bash
pip install numpy torch matplotlib jupyter transformers
```

Tested on Python 3.10+, PyTorch 2.0+. CPU is fine — all four parts run in seconds to a few minutes.

**Note on `transformers`:** the first run of Part 4 downloads `distilbert-base-uncased-finetuned-sst-2-english` (~250 MB). Subsequent runs use the cached weights.

## Data

`data/dr_seuss.txt` — small corpus (2,071 tokens, ~308 unique words) used for the Word2Vec exercise in Part 3.

## Running a notebook

```bash
cd homework_3/code
jupyter notebook homework_3_solution.ipynb       # answer key
# or:
jupyter notebook homework_3_student.ipynb        # student version
```

Run cells top-to-bottom. Outputs are reproducible (`np.random.seed(42)`, `torch.manual_seed(42)`).

## Editing

To change any cell:

1. Edit the corresponding entry in `_build_notebook.py` (the `CELLS` list).
2. Run `python _build_notebook.py` to regenerate both notebooks.

Three cell helpers:

- `md("...")` — markdown cell, identical in both versions.
- `code("...")` — code cell, identical in both versions (setup, plotting, verification).
- `solution(student_stub, full_solution)` — code cell with two sources. The solution notebook gets `full_solution`; the student notebook gets `student_stub` (a TODO scaffold).

## Section map

The notebook has four parts, mirroring Section 4 of the PDF:

| Part | Problem | Topic | What students implement |
|------|---------|-------|--------------------------|
| 1 | 17(a) | MF with SGD on synthetic panel data | `train_mf_sgd()`: gradients, in-loop loss tracking, early stopping |
| 1 | 17(b) | Rank sweep | Loop over $K \in \{1, 2, 3, 5, 8, 12\}$, compute $E_\text{in}, E_\text{out}$, plot |
| 1 | 17(c) | Best K | Identify $K^* = \arg\min E_\text{out}$ |
| 1 | 17(d) | Regularization sweep | Loop $\lambda \in \{0, 10^{-3}, 10^{-2}, 10^{-1}\}$ over the same $K$ grid |
| 2 | 18(a) | `SelfAttention` | scaled dot-product attention, verify match with `F.scaled_dot_product_attention` |
| 2 | 18(b) | `MultiHeadAttention` | $h = 4$ heads, output projection $\mathbf{W}_O$ |
| 3 | 19(a) | `generate_traindata()` | sliding-window pair generation, one-hot encoding |
| 3 | 19(b) | Skip-gram network | 2-layer linear net, Adam, cross-entropy, extract embeddings |
| 3 | 19(c) | Top-30 most similar pairs | cosine similarity, dedup, sort |
| 4 | 20(a) | Pretrained DistilBERT sentiment | inference on 5 finance sentences |
| 4 | 20(b) | Attention visualization | extract `outputs.attentions[3][0, 0]`, plot heatmap |

## Student notebook: what's stripped

The following cells are replaced with `TODO` scaffolds in the student version:

| Cell | Function/class |
|------|----------------|
| Problem 17(a) | `train_mf_sgd()` |
| Problem 17(b) | rank-sweep loop and plot |
| Problem 17(d) | regularization-sweep loop |
| Problem 18(a) | `SelfAttention` class |
| Problem 18(b) | `MultiHeadAttention` class |
| Problem 19(a) | `one_hot()` + `generate_traindata()` |
| Problem 19(b) | skip-gram model + training loop + embedding extraction |
| Problem 19(c) | `top_similar_pairs()` |
| Problem 20(b) | attention extraction + heatmap |

Setup cells (data generation, verification, plotting scaffolds, model loading) are identical in both versions.

## Synthetic panel data details

The data-generating process simulates an econ panel:

- 20 firms × 12 months of "log sales"
- True low-rank factor model with $K^* = 3$ (e.g., industry, regional, macro factors)
- Gaussian noise, $\sigma = 0.2$
- 25% of entries masked uniformly at random (think: staggered treatment timing)

The exercise mirrors **Athey, Bai, Imbens, Khosravi (2021)** — recover counterfactual outcomes for treated cells via matrix completion.
