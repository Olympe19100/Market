from abc import ABC, abstractmethod
from typing import Any

class Metric(ABC):
    """
    Abstract base class for all metrics.
    """
    
    @abstractmethod
    def calculate(self, *args, **kwargs) -> Any:
        """
        Calculates the metric.
        """
        pass
