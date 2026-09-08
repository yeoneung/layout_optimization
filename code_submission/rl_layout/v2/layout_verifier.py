"""Independent verification for complete fixed-shape layouts.

The verifier intentionally does not import the objective, a constructor, or a
completion generator.  Its trusted input is only the immutable instance
geometry and a list of ``(facility, row, column)`` placements.  This keeps the
feasibility guarantee independent of every scoring and search implementation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Verification:
    valid: bool
    reason: str = "ok"


def _integer(value):
    """Return an exact Python integer, or ``None`` for non-integral input."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != float(integer):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return integer


def verify_complete_layout(iw, ih, gw, gh, placements, prefix=()):
    """Verify identities, dimensions, containment, nonoverlap and a prefix.

    ``iw`` and ``ih`` are the fixed integer dimensions indexed by facility id.
    ``gw`` and ``gh`` are the plate dimensions.  ``prefix`` is an optional list
    of earlier commitments that the returned layout must reproduce exactly.
    The function returns a diagnostic ``Verification`` instead of raising, so
    failed candidates can be discarded safely by a search procedure.
    """
    iw = np.asarray(iw)
    ih = np.asarray(ih)
    if iw.ndim != 1 or ih.ndim != 1 or len(iw) != len(ih):
        return Verification(False, "dimension-vector shape")
    n = len(iw)
    gw_i, gh_i = _integer(gw), _integer(gh)
    if gw_i is None or gh_i is None or gw_i <= 0 or gh_i <= 0:
        return Verification(False, "plate dimensions")
    if any(_integer(v) is None or _integer(v) <= 0 for v in iw):
        return Verification(False, "facility widths")
    if any(_integer(v) is None or _integer(v) <= 0 for v in ih):
        return Verification(False, "facility heights")
    if placements is None:
        return Verification(False, "missing placement list")
    try:
        placements = list(placements)
    except TypeError:
        return Verification(False, "placement list")
    if len(placements) != n:
        return Verification(False, "missing or extra facility")

    parsed = {}
    for item in placements:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            return Verification(False, "placement record")
        facility, row, column = map(_integer, item)
        if facility is None or row is None or column is None:
            return Verification(False, "non-integral placement")
        if facility < 0 or facility >= n:
            return Verification(False, "facility id out of range")
        if facility in parsed:
            return Verification(False, "duplicate facility")
        parsed[facility] = (row, column)
    if set(parsed) != set(range(n)):
        return Verification(False, "missing facility")

    expected_prefix = {}
    try:
        prefix_items = list(prefix)
    except TypeError:
        return Verification(False, "prefix list")
    for item in prefix_items:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            return Verification(False, "prefix record")
        facility, row, column = map(_integer, item)
        if facility is None or row is None or column is None:
            return Verification(False, "non-integral prefix")
        if facility in expected_prefix:
            return Verification(False, "duplicate prefix facility")
        expected_prefix[facility] = (row, column)
    for facility, position in expected_prefix.items():
        if facility not in parsed or parsed[facility] != position:
            return Verification(False, "stale or inconsistent prefix")

    occupied = np.full((gh_i, gw_i), -1, dtype=np.int64)
    for facility in range(n):
        row, column = parsed[facility]
        width, height = int(iw[facility]), int(ih[facility])
        if row < 0 or column < 0 or row + height > gh_i or column + width > gw_i:
            return Verification(False, "out of bounds")
        window = occupied[row:row + height, column:column + width]
        if np.any(window >= 0):
            return Verification(False, "overlap")
        window[:] = facility
    return Verification(True)


def verify_witness_completion(iw, ih, gw, gh, committed, witness):
    """Verify that ``witness`` completes the exact committed prefix."""
    try:
        complete = list(committed) + list(witness)
    except TypeError:
        return Verification(False, "missing prefix or witness")
    return verify_complete_layout(iw, ih, gw, gh, complete, prefix=committed)

