# Notebook Workflow

Use this folder as the home for ad hoc cloud-notebook experiments.

The recommended workflow is:

1. Start a Linux GPU notebook.
2. Bring the repository into the runtime.
3. Install dependencies.
4. Run `python scripts/show_runtime_info.py`.
5. Run `python scripts/run_smoke_suite.py`.
6. Launch the long training jobs from notebook cells.

Minimal notebook bootstrap:

```python
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
!python scripts/show_runtime_info.py
!pytest -q -s tests/test_env.py tests/test_graph.py
!python scripts/run_smoke_suite.py
```

For the full cloud workflow, follow [RUN_GUIDE.md](../RUN_GUIDE.md).
