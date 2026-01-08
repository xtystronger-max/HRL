class HRL3DError(Exception):
    """Base error."""


class ValidationError(HRL3DError):
    """Raised when inputs or plans are invalid."""


class PlanningError(HRL3DError):
    """Raised when a planner cannot find a feasible plan."""


class OptionalDependencyError(HRL3DError):
    """Raised when an optional dependency is required but missing."""
