#!/usr/bin/env python3

from functools import lru_cache
import re
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..correlator import CorrelatorEnsemble, Correlator


def pair_to_complex(value):
    real, imag = map(float, value.strip("()").split(","))
    return real + imag * 1J


def read_single_mres_file(filepath, ensemble):
    cfg_index = int(re.search("([0-9]+).xml$", filepath.name).groups()[0])
    for elem in ET.parse(filepath).getroot()[0]:
        if elem.tag == "mass":
            valence_mass = float(elem.text)
            continue

        channel_name = elem.tag

        values = []
        for value_elem in elem:
            values.append(pair_to_complex(value_elem.text))

        ensemble.append(
            Correlator(
                "run1",
                cfg_index,
                "DEFAULT_SEMWALL",
                "TRIPLET",
                channel_name,
                valence_mass,
                np.asarray(values),
            )
        )


def read_single_meson_file(filepath, ensemble):
    cfg_index = int(re.search("([0-9]+).xml$", filepath.name).groups()[0])
    for corr in ET.parse(filepath).getroot()[0]:
        values = []
        source = None
        sink = None
        for elem in corr:
            if elem.tag == "gamma_src":
                source = elem.text
            elif elem.tag == "gamma_snk":
                sink = elem.text
            elif elem.tag == "corr":
                for value_elem in elem:
                    values.append(pair_to_complex(value_elem.text))
            else:
                logging.warn(f"Element <{elem.tag}> not understood.")

            channel_name = f"{source}_{sink}"

        ensemble.append(
            Correlator(
                "run1",
                cfg_index,
                "DEFAULT_SEMWALL",
                "TRIPLET",
                channel_name,
                ensemble.metadata.get("fermion_mass"),
                np.asarray(values),
            )
        )


@lru_cache(maxsize=8)
def read_correlators_hadrons(directory, **metadata):
    directory = Path(directory)

    correlators = CorrelatorEnsemble(directory)
    correlators.metadata = metadata

    for filepath in directory.iterdir():
        if filepath.name.startswith("mres"):
            read_single_mres_file(filepath, correlators)
        elif filepath.name.startswith("pt_ll"):
            read_single_meson_file(filepath, correlators)

    correlators.freeze()

    if not correlators.is_consistent:
        logging.warning("Correlator is not self-consistent.")

    return correlators
