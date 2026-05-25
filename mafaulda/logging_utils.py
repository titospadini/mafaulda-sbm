import sys

# Global verbosity level
# 0: Quiet/Silent (only error stacktraces or explicit forced prints)
# 1: Standard/Default (current print levels)
# 2: Detailed/Verbose (more details, timings, etc.)
# 3: Debug (matrix shapes, sanity check passing states, etc.)
_verbosity = 1

def set_verbosity(level: int) -> None:
    """Sets the global verbosity level."""
    global _verbosity
    _verbosity = level

def get_verbosity() -> int:
    """Returns the current global verbosity level."""
    return _verbosity

def log(msg: str = "", level: int = 1, **kwargs) -> None:
    """
    Prints a message conditionally if the current verbosity is >= the specified level.
    """
    if _verbosity >= level:
        print(msg, **kwargs)
