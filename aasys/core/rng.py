"""Seeded random number generation.

Every stochastic component draws from a named substream so that adding or
removing a sensor does not perturb the noise seen by unrelated components.
That keeps A/B comparisons (e.g. radar-only vs fused) genuinely controlled.
"""

from __future__ import annotations

import hashlib

import numpy as np


class RngHub:
    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        """Return a stable, independent generator for `name`."""
        if name not in self._streams:
            digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
            offset = int.from_bytes(digest, "little")
            self._streams[name] = np.random.default_rng(
                (self.seed + offset) % (2**63)
            )
        return self._streams[name]
