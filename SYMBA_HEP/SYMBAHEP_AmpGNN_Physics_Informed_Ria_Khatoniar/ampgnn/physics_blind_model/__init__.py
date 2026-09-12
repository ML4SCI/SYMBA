"""Physics-blind baseline for comparison against the physics-informed AmpGNN.

Self-contained: nothing in the parent package is modified. Shared, physics-neutral
machinery (data loading, train/val/test split, target tokenisation, vocabulary,
decoder, losses, metrics) is imported from the parent so the two models are
compared on exactly the same footing.
"""
