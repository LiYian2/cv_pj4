from __future__ import annotations

from typing import List


def build_micro_gaussian_param_groups(gaussians, mode: str, lr_xyz: float, lr_opacity: float) -> List[dict]:
    mode = (mode or 'xyz').strip().lower()
    if mode not in {'xyz', 'xyz_opacity'}:
        raise ValueError(f'Unsupported micro-gaussian mode: {mode}')

    groups = [
        {
            'params': [gaussians._xyz],
            'lr': float(lr_xyz),
            'name': 'xyz',
        }
    ]
    if mode == 'xyz_opacity':
        groups.append(
            {
                'params': [gaussians._opacity],
                'lr': float(lr_opacity),
                'name': 'opacity',
            }
        )
    return groups
