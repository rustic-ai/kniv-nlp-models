"""kniv v6 — next-generation training pipeline.

Deliberately separate from the v5 work under ``models/``, ``shared/`` and
``scripts/``. Nothing here imports from those packages; v5 is referenced
only as a *teacher* (an annotator), never as a code dependency.

Current contents: the annotator bake-off (``v6.bakeoff``), which measures
each candidate LLM annotator against public gold so that the
"which annotator owns which layer" decision is a number, not a guess.
"""

__version__ = "0.1.0"
