from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

ALLOWED_SIMULATION_STATUSES = frozenset({429, 500, 502, 503})


@dataclass(frozen=True)
class SimulationPlan:
    """Per-request failure simulation rules parsed from X-Simulate-Failure."""

    rules: Mapping[str, int]

    @property
    def is_active(self) -> bool:
        return bool(self.rules)

    def status_for_key(self, key_id: str) -> Optional[int]:
        """Return simulated HTTP status code for key_id, or None if not simulated."""
        if key_id in self.rules:
            return self.rules[key_id]
        return self.rules.get("*")


def parse_failure_simulation_header(
    header_value: Optional[str], enabled: bool
) -> SimulationPlan:
    """Parse X-Simulate-Failure header safely when simulation is enabled.

    Unknown tokens, malformed pairs, or unsupported status codes are ignored
    rather than crashing.
    """
    if not enabled or not header_value:
        return SimulationPlan(rules={})

    rules: dict[str, int] = {}
    for token in header_value.split(","):
        cleaned = token.strip()
        if "=" not in cleaned:
            continue
        key_part, status_part = cleaned.split("=", 1)
        key_id = key_part.strip()
        status_str = status_part.strip()
        if not key_id or not status_str.isdigit():
            continue
        status_code = int(status_str)
        if status_code not in ALLOWED_SIMULATION_STATUSES:
            continue
        rules[key_id] = status_code

    return SimulationPlan(rules=rules)
