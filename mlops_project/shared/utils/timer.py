"""
Timer utility for measuring execution time.
"""
import time
from typing import Optional
from .logging_utils import get_logger

logger = get_logger(__name__)


class Timer:
    """
    Context manager for timing code execution.
    
    Usage:
        with Timer("training"):
            # code to time
    """
    
    def __init__(self, name: Optional[str] = None, log: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Name for the timed operation
            log: Whether to log the elapsed time
        """
        self.name = name or "operation"
        self.log = log
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        if self.log:
            logger.info(f"{self.name} completed in {self.elapsed:.4f} seconds")
