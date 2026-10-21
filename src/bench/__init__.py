"""Fine-tuning benchmark harness: one model registry, one task registry.

A run is identified by (model, task, arm, size, seed, lr) and nothing else, and
every model-specific or task-specific fact lives in exactly one place:

    models.py          what a model is: prompt, stop token, adapter targets
    tasks/__init__.py  what a task is: splits, arms, encoding, scoring

The PIZZA task's dataset is CC-BY-NC-4.0. Neither the data, the predictions nor
any adapter trained on it may be used by the production cafe parser or committed
to git.

The boundary is kept by the registry loading tasks lazily: importing
`bench.tasks.coffee`, which is the shipped cafe parser, never imports a single
line of PIZZA code, and the cafe task uses nothing but the standard library.
`tests/test_registries.py` asserts that rather than trusting it. See PROTOCOL.md.
"""

from . import config, models, tasks

__all__ = ["config", "models", "tasks"]
