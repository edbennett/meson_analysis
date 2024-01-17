#!/usr/bin/env python3

from .read_flexlatsim import read_correlators_flexlatsim
from .read_hadrons import read_correlators_hadrons
from .read_hirep import read_correlators_hirep


readers = {
    "flexlatsim": read_correlators_flexlatsim,
    "hadrons": read_correlators_hadrons,
    "hirep": read_correlators_hirep,
}
