"""a100-log-buffer-budget: offline burstiness and buffer-budget analysis for
CAN-FD logs recorded alongside RadarTools target logs.

The package is deliberately offline and statistical: it parses Vector ASC
logs, characterises multi-scale arrival-rate burstiness, and simulates a
deterministic in-memory consumer queue. It never decodes protocol payload
semantics and never touches a live bus.
"""

__version__ = "1.0.0"

__all__ = ["asc_parser", "bursts", "queue_sim", "cli"]
