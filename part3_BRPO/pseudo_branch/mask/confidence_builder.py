
# -*- coding: utf-8 -*-
"""Build simplified confidence mask."""
import numpy as np

def build_confidence_from_target_depth(target_depth, render_depth, threshold=0.1):
    """Simplified confidence: 1 where target_depth > 0."""
    confidence = (target_depth > 0).astype(np.float32)
    return confidence
