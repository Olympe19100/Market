class MarketMakerError(Exception):
    """Base exception for market maker errors."""
    pass

class ModelError(MarketMakerError):
    """Raised when there is an error in the model."""
    pass

class OrderExecutionError(MarketMakerError):
    """Raised when an order fails to execute."""
    pass

class InsufficientBalanceError(MarketMakerError):
    """Raised when there is insufficient balance."""
    pass

class InvalidStateError(MarketMakerError):
    """Raised when the market state is invalid."""
    pass

class NetworkError(MarketMakerError):
    """Raised when there are network issues."""
    pass

class DataError(MarketMakerError):
    """Base exception class for data related errors."""
    pass

class ProcessingError(DataError):
    """Raised when there is an error processing data."""
    pass

class PositionError(MarketMakerError):
    """Raised when there is a position-related error."""
    pass
