"""Project-wide conventions.

OUTCOMES fixes the column order of every probability array in this project:
column 0 = home win, column 1 = draw, column 2 = away win.

Note: sklearn sorts class labels alphabetically ("A", "D", "H"), which is the
REVERSE of this order. Never pass an (H, D, A) array to an sklearn metric
without reordering; use eplmodel.evaluation.metrics instead, which takes the
column order explicitly.
"""

OUTCOMES = ("H", "D", "A")

# Elo "actual score" convention: a draw counts as half a win.
RESULT_TO_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}
