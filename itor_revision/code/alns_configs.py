"""Extended ALNS configuration grid for the retuning study (X3).

The archive's repair_baseline.CONFIGURATIONS holds the four published
candidates.  This module adds a structured grid without modifying the archive:
removal cap {4, 8, 16, 32} x beam width {4, 16} x temperature fraction
{0.0025, 0.01}, all with four ranked candidate positions, and registers the
entries in the archive dictionary at import time (process-wide, recorded in
the study protocol).  Names are stable identifiers of the form
alns_c<cap>_b<beam>_t<temperature x 10000>.
"""

import repair_baseline

GRID = {}
for cap in (4, 8, 16, 32):
    for beam in (4, 16):
        for temperature in (0.0025, 0.01):
            name = "alns_c{}_b{}_t{}".format(cap, beam, int(round(temperature * 10000)))
            GRID[name] = {"destroy_cap": cap, "beam_width": beam, "candidates": 4,
                          "temperature_fraction": temperature}


def register():
    """Add the grid to repair_baseline.CONFIGURATIONS; published entries are untouched."""
    for name, config in GRID.items():
        repair_baseline.CONFIGURATIONS.setdefault(name, dict(config))
    return sorted(GRID)


register()
