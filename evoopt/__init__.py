"""
EvoOpt-Surrogate-Bench: Constrained Engineering Optimization & Surrogate-Assisted
Evolutionary Algorithms (SAEA) Framework.
"""

import platform

if hasattr(platform, "_wmi"):
    platform._wmi = None

from evoopt.problems import (
    BenchmarkProblem,
    MultiObjectiveProblem,
    ThreeBarTruss,
    PressureVessel,
    WeldedBeam,
    TensionCompressionSpring,
    ChemicalReactionKineticsMOO,
    AOPOzonationKineticsMOO,
    ZDT1,
    ZDT2,
    ZDT3,
    OSY,
    get_problem,
)
from evoopt.constraints import (
    compute_cv,
    ConstraintHandler,
    DebFeasibilityRules,
    AdaptivePenaltyConstraintHandler,
    EpsilonConstraintHandler,
    get_constraint_handler,
)
from evoopt.optimizers import (
    OptimizationResult,
    MultiObjectiveResult,
    HEOA,
    DifferentialEvolution,
    DE,
    ParticleSwarmOptimization,
    PSO,
    NSGA2,
    MOEAD,
    fast_non_dominated_sort,
    calculate_crowding_distance,
)
from evoopt.surrogate import (
    probability_of_feasibility,
    expected_improvement,
    constrained_expected_improvement,
    feasibility_aware_lcb,
    GaussianProcessSurrogate,
    DeepEnsembleSurrogate,
    SurrogateAssistedHEOA,
    SurrogateAssistedNSGA2,
)
from evoopt.metrics import (
    compute_hypervolume,
    compute_igd,
    find_knee_point,
    is_pareto_efficient,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkProblem",
    "MultiObjectiveProblem",
    "ThreeBarTruss",
    "PressureVessel",
    "WeldedBeam",
    "TensionCompressionSpring",
    "ChemicalReactionKineticsMOO",
    "AOPOzonationKineticsMOO",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "OSY",
    "get_problem",
    "compute_cv",
    "ConstraintHandler",
    "DebFeasibilityRules",
    "AdaptivePenaltyConstraintHandler",
    "EpsilonConstraintHandler",
    "get_constraint_handler",
    "OptimizationResult",
    "MultiObjectiveResult",
    "HEOA",
    "DifferentialEvolution",
    "DE",
    "ParticleSwarmOptimization",
    "PSO",
    "NSGA2",
    "MOEAD",
    "fast_non_dominated_sort",
    "calculate_crowding_distance",
    "probability_of_feasibility",
    "expected_improvement",
    "constrained_expected_improvement",
    "feasibility_aware_lcb",
    "GaussianProcessSurrogate",
    "DeepEnsembleSurrogate",
    "SurrogateAssistedHEOA",
    "SurrogateAssistedNSGA2",
    "compute_hypervolume",
    "compute_igd",
    "find_knee_point",
    "is_pareto_efficient",
]
