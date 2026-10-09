"""v6 training: one encoder, every head, one pass over a window.

Design and operations are in ``v6/TRAINING.md``. Nothing here imports from
the v5 code under ``models/`` or ``shared/``; the head shapes follow v5's
conventions but are reimplemented.
"""
