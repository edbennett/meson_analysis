#!/usr/bin/env python3

from functools import lru_cache
import re
import logging

import numpy as np

from ..correlator import CorrelatorEnsemble, Correlator


_reps = {
    "FUN": "fundamental",
    "SYM": "symmetric",
    "ASY": "antisymmetric",
    "ADJ": "adjoint",
}


def get_representation(macros):
    macros = [macros[0].replace("[SYSTEM][0]MACROS=", "")] + macros[1:]

    for macro in macros:
        if macro.startswith("-DREPR_NAME"):
            return macro.replace('-DREPR_NAME="REPR_', "").strip('"').lower()


def add_metadata(metadata, line_contents):
    """
    Parse out possible metadata given on a line,
    and add it to an ensemble metadata dictionary.
    """
    if (
        line_contents[0] == "[GEOMETRY][0]Global"
        or line_contents[0] == "[GEOMETRY_INIT][0]Global"
        or line_contents[0] == "[MAIN][0]global"
    ):
        NT, NX, NY, NZ = map(
            int,
            re.match("([0-9]+)x([0-9]+)x([0-9]+)x([0-9]+)", line_contents[3]).groups(),
        )
        metadata["NT"] = NT
        metadata["NX"] = NX
        metadata["NY"] = NY
        metadata["NZ"] = NZ

    if line_contents[0].startswith("[MAIN][0]Mass["):
        if "valence_masses" not in metadata:
            metadata["valence_masses"] = []
        if (valence_mass := float(line_contents[2])) not in metadata["valence_masses"]:
            metadata["valence_masses"].append(valence_mass)

    if line_contents[:2] == ["[MAIN][0]Fermion", "representation:"]:
        metadata["valence_representation"] = line_contents[2][5:].lower()

    if line_contents[1] == "group:" and line_contents[0] in [
        "[SYSTEM][0]Gauge",
        "[MAIN][0]Gauge",
    ]:
        group_family, Nc = line_contents[2].strip(")").split("(")
        metadata["group_family"] = group_family
        metadata["Nc"] = int(Nc)

    if line_contents[0].startswith("[SYSTEM][0]MACROS="):
        metadata["valence_representation"] = get_representation(line_contents)


def add_single_consistent_metadatum(metadata, name, value):
    if value and name in metadata and metadata[name] != value:
        logging.warning(f"{name} values are not consistent!")
    metadata[name] = value


def add_cfg_metadata(metadata, Nc, rep, Nf, beta, mass):
    """
    Verify and attach metadata from analysed configurations
    to an ensemble metadata dictionary.
    """
    if int(Nc) != metadata.get("Nc"):
        logging.warning("Configuration Nc does not match valence Nc")

    add_single_consistent_metadatum(metadata, "dynamical_representation", rep)
    add_single_consistent_metadatum(metadata, "Nf", Nf)
    add_single_consistent_metadatum(metadata, "beta", beta)
    add_single_consistent_metadatum(metadata, "dynamical_mass", mass)


def parse_cfg_filename(filename):
    """
    Parse out the run parameters and trajectory index from a configuration filename.

    Arguments:

        filename: The configuration filename

    Returns:

        run_name: The name of the run/stream
        Nc: the number of colors
        rep: the fermion representation used for dynamical flavors
        Nf: the number of dynamical fermion flavors
        beta: the lattice coupling beta used to generate the ensemble
        mass: the mass of dynamical fermion flavors
        cfg_index: The index of the trajectory in the stream
    """

    matched_filename = re.match(
        r".*/([^/]*)_[0-9]+x[0-9]+x[0-9]+x[0-9]+nc([0-9]+)(?:r([A-Z]+))?(?:nf([0-9]+))?b([0-9]+\.[0-9]+)m(-?[0-9]+\.[0-9]+)n([0-9]+)",
        filename,
    )
    run_name, Nc, rep, Nf, beta, mass, cfg_index = matched_filename.groups()

    return (
        run_name,
        int(Nc),
        _reps[rep] if rep else None,
        int(Nf),
        float(beta),
        float(mass),
        int(cfg_index),
    )


def add_row(ensemble, split_line, stream_name, cfg_index):
    """
    Add a single correlation function measurement to the ensemble.

    Arguments:

        ensemble: The CorrelatorEnsemble to append to.
        split_line: A list of strings, the split line of the input data.
        stream_name: The identifier of the Monte Carlo stream from
                     which the data are taken.
        cfg_index: The index of the configuration being measured on
                   within its Monte Carlo stream.
    """
    valence_mass = float(split_line[2][5:])
    try:
        _ = float(split_line[5])
    except ValueError:
        # Column 5 is a channel name, so source type is explicit
        source_type = split_line.pop(3)
    else:
        # Column 5 is a number, so source type is not stated
        source_type = "DEFAULT_SEMWALL"

    connection_type = split_line[3]
    channel = split_line[4][:-1]
    correlator = np.asarray(split_line[5:], dtype=float)

    ensemble.append(
        Correlator(
            stream_name,
            cfg_index,
            source_type,
            connection_type,
            channel,
            valence_mass,
            correlator,
        )
    )


@lru_cache(maxsize=8)
def read_correlators_hirep(filename):
    """
    Read the correlation functions and associated metadata
    from the file in the specified `filename`.
    """
    correlators = CorrelatorEnsemble(filename)

    # Track which configurations have provided results;
    # don't allow duplicates
    read_cfgs = set()
    run_repr = None

    with open(filename) as f:
        for line in f.readlines():
            line_contents = line.split()
            if not line_contents:
                continue
            if line[0].startswith("[SYSTEM][0]MACROS="):
                for flag in line:
                    if flag.startswith("-DREPR_"):
                        run_repr = flag[7:].lower()
                        continue
                else:
                    run_repr = None
            if (
                line_contents[0] == "[IO][0]Configuration"
                and line_contents[2] == "read"
            ):
                run_name, Nc, rep, Nf, beta, mass, cfg_index = parse_cfg_filename(
                    line_contents[1][1:-1]
                )
                if rep is None:
                    rep = run_repr
                elif run_repr is not None and repr != run_repr:
                    raise ValueError(
                        "Representation mismatch between ensemble and code"
                    )

                add_cfg_metadata(correlators.metadata, Nc, rep, Nf, beta, mass)

                if (run_name, cfg_index) in read_cfgs:
                    logging.warn(
                        f"Possible duplicate data in {run_name} trajectory {cfg_index} of file {filename}"
                    )
                continue

            add_metadata(correlators.metadata, line_contents)

            if line_contents[0] == "[MAIN][0]conf":
                read_cfgs.add((run_name, cfg_index))
                add_row(correlators, line_contents, run_name, int(cfg_index))

    correlators.freeze()

    if not correlators.is_consistent:
        logging.warning("Correlator is not self-consistent.")

    return correlators
