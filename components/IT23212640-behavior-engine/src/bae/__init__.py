"""Behavioral Analytics Engine (BAE).

Behavior-Based Threat Intelligence Engine for real-time threatening behaviour and
repeated-harassment detection. Component of project J26-CS-337 (SLIIT, IAS cluster).
Author: Jayarathna W.D.O.B. (IT23212640)
"""

__version__ = "1.0.0"

# Windows consoles default to a legacy code page; never crash on a non-ASCII character in output.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass
