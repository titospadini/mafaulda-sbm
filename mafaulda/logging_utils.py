import sys

# Ensure UTF-8 output across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
        try:
            print(msg, **kwargs)
        except UnicodeEncodeError:
            clean_msg = msg.encode("ascii", errors="replace").decode("ascii")
            print(clean_msg, **kwargs)

