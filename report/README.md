# Report Assets

- Main draft: [paper.md](paper.md)
- LaTeX source: [paper.tex](paper.tex)
- Figures: `report/figures/`

For the real course experiments, the canonical artifact path is now the McGill `mimi` + Slurm DAG:

- experiment checkpoints and summaries under `artifacts/experiments/`
- final evaluation summaries from `scripts/postprocess_experiments.py`
- figures regenerated into `report/figures/`

Use local Pixi or Tectonic primarily for PDF compilation after those artifacts exist.

You can build the paper with Tectonic:

```bash
tectonic report/paper.tex
```

If you are using Pixi:

```bash
pixi run paper
```

The Markdown draft and the LaTeX source contain the same project content. Use `paper.tex` for PDF submission and `paper.md` for easier editing.
