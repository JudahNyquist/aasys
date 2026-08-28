"""Post-run analysis: recording, scoring and plotting.

This package is allowed to read ground truth (see `core.truth_guard`) for
the same reason `lethality` is: it reports *about* a run rather than feeding
anything back into it.
"""

from .recorder import Recorder

__all__ = ["Recorder"]
