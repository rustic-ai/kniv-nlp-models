"""Gold-standard loaders.

Public benchmarks are used here for **evaluation only** — never as v6
training data. This is what keeps the bake-off falsifiable: the corpus is
ours, the yardstick is not.
"""
from .ud_ewt import load_ud_items          # noqa: F401
from .propbank import load_srl_items       # noqa: F401
from .ner import load_ner_items, map_to_conll   # noqa: F401
from .redocred import load_rel_items, relation_inventory   # noqa: F401
