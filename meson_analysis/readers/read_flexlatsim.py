#!/usr/bin/env python3

from functools import lru_cache
import logging
import warnings

import numpy as np

from ..correlator import CorrelatorEnsemble, Correlator


channels = {
    "AA": "AA",
    "AP": "AP",
    "AV0": "AV0",
    "V0P": "V0P",
    "V0V0": "V0V0",
    "pscalar": "g5",
    "pvector": "pvector",
    "scalar": "id",
    "vector": "gk",
    "dsapcorr": "dsapcorr",
    "GGCorrTr1": "GGCorrTr1",
    "GGCorrTr4": "GGCorrTr4",
    "WICorr1tr1": "WICorr1tr1",
    "WICorr1tr4": "WICorr1tr4",
    "WICorr2tr1": "WICorr2tr1",
    "WICorr2tr4": "WICorr2tr4",
    "WICorr3tr1": "WICorr3tr1",
    "WICorr3tr4": "WICorr3tr4",
}


def add_row(
    ensemble, read_cfgs, trajectory, state, stream_name, valence_mass, raw_correlator
):
    read_cfgs.add((stream_name, trajectory, state))
    ensemble.append(
        Correlator(
            stream_name,
            trajectory,
            "POINT",  # TODO CHECK
            "TRIPLET",  # TODO CHECK
            channels[state],
            valence_mass,
            np.asarray(raw_correlator),
        )
    )


@lru_cache
def read_correlators_flexlatsim(
    filename, valence_mass, correlators=None, stream_name="", freeze=True, **metadata
):
    if correlators is None:
        correlators = CorrelatorEnsemble(filename)
    elif correlators.frozen:
        raise ValueError("Can't load extra data to a frozen CorrelatorEnsemble")

    # Track which configurations have provided results;
    # don't allow duplicates
    read_cfgs = set(
        [
            (corr.stream_name, corr.cfg_index, corr.channel)
            for corr in correlators.correlators
        ]
    )

    correlators.metadata.update(metadata)

    with open(filename) as f:
        current_trajectory = None
        current_state = None
        correlator_values = []

        for line in f.readlines():
            if not line.startswith("(MM):"):
                continue
            split_line = line.split(":")
            if len(split_line) < 3:
                continue
            if split_line[2] not in ("Meson_corr", "gluinoglue"):
                continue
            if split_line[3] in (
                "total_inviter",
                "source_t",
                "leveli",
                "levelj",
                "itpropagator",
                "JN1",
                "source",
            ):
                continue
            if "(MM)" in line[5:]:
                warnings.warn(f"Corrupt line found, skipping:\n{line}")
                continue

            new_trajectory = int(split_line[1])
            new_state = split_line[3]

            if new_trajectory != current_trajectory:
                if correlator_values:
                    add_row(
                        correlators,
                        read_cfgs,
                        current_trajectory,
                        current_state,
                        stream_name,
                        valence_mass,
                        correlator_values,
                    )
                current_trajectory = new_trajectory
                t_index = 0
                correlator_values = []

            if new_state != current_state:
                if correlator_values:
                    add_row(
                        correlators,
                        read_cfgs,
                        current_trajectory,
                        current_state,
                        stream_name,
                        valence_mass,
                        correlator_values,
                    )

                current_state = new_state
                t_index = 0
                correlator_values = []

            if (stream_name, new_trajectory, new_state) in read_cfgs:
                warnings.warn(
                    f"Possible duplicate data in {stream_name} trajectory {new_trajectory} of file {filename}"
                )
                continue

            if split_line[3].startswith("GGCorrTr"):
                time_index, value_index = 8, 10
            elif split_line[3].startswith("WICorr"):
                time_index, value_index = 5, 7
            else:
                time_index, value_index = 7, 9

            try:
                int(split_line[time_index])
            except:
                breakpoint()
            if int(split_line[time_index]) != t_index:
                raise ValueError(
                    f"Trajectory {current_trajectory}, state {current_state} inconsistent at time index {t_index} ≠ {split_line[time_index]}"
                )

            correlator_values.append(float(split_line[value_index]))
            t_index += 1
        else:
            if correlator_values:
                add_row(
                    correlators,
                    read_cfgs,
                    current_trajectory,
                    current_state,
                    stream_name,
                    valence_mass,
                    correlator_values,
                )

    if freeze:
        correlators.freeze()
        if not correlators.is_consistent:
            logging.warning("Correlator is not self-consistent.")

    return correlators
