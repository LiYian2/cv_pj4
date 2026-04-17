from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PseudoLocalGatingConfig:
    mode: str = 'off'
    params: str = 'xyz'
    min_verified_ratio: float = 0.01
    min_rgb_mask_ratio: float = 0.01
    max_fallback_ratio: float = 0.995
    min_correction: float = 0.0
    soft_power: float = 1.0
    log_interval: int = 20

    def enabled(self) -> bool:
        return (self.mode or 'off') != 'off'

    def is_soft(self) -> bool:
        return (self.mode or '').startswith('soft_')

    def as_dict(self) -> dict:
        return asdict(self)
