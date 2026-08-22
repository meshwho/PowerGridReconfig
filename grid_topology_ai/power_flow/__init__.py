from enum import StrEnum


class PowerFlowError(RuntimeError):
    pass


class PowerFlowNotConverged(PowerFlowError):
    pass


class InvalidPhysicalState(PowerFlowError):
    pass


class PowerFlowFailureKind(StrEnum):
    NOT_CONVERGED = "not_converged"
    INVALID_PHYSICAL_STATE = "invalid_physical_state"
