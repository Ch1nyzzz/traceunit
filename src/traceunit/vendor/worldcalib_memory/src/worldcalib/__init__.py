"""WorldCalib baseline package.

Carved out of Optimizer1 to keep only what is needed to run LOCOMO and
LongMemEval baseline evaluations. The Python package name remains
``worldcalib`` so that internal imports keep working without rewrites.
"""

from worldcalib.evaluation import EvaluationRunner, run_initial_frontier
from worldcalib.pareto import ParetoPoint, pareto_frontier

__all__ = [
    "EvaluationRunner",
    "ParetoPoint",
    "pareto_frontier",
    "run_initial_frontier",
]
