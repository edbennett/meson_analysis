#!/usr/bin/env python3

from collections import namedtuple
import logging

import numpy as np
import pandas as pd
import pyerrors as pe


class CorrelatorEnsemble:
    """
    Represents a full ensemble of correlation functions.
    """

    _frozen = False

    def __init__(self, filename):
        self.correlators = []
        self.metadata = {}
        self.filename = filename

    def append(self, correlator):
        """
        Append a single Correlator instance to the ensemble.
        """

        if self._frozen:
            raise ValueError("Can't append to a frozen CorrelatorEnsemble.")

        if self.correlators and len(correlator.correlator) != len(
            self.correlators[0].correlator
        ):
            raise ValueError("Correlation function lengths don't match.")

        self.correlators.append(
            correlator._replace(correlator=np.asarray(correlator.correlator))
        )

    def freeze(self):
        """
        Turn the list of correlators into a Pandas DataFrame
        to ease subsequent processing.
        """

        self.correlators = pd.DataFrame(self.correlators).sort_values(
            ["stream_name", "cfg_index", "source_type", "connection_type"]
        )
        self._frozen = True

    def get(self, **criteria):
        """
        Get the correlators for a specific channel as a Pandas Series.

        Arguments:
            **criteria: Fields to filter on, and the filter values.
        """
        if not self._frozen:
            raise ValueError("Need to be frozen to subset the ensemble.")

        subset = self.correlators.copy()
        for field, value in criteria.items():
            subset = subset[subset[field] == value]

        for field in "source_type", "connection_type", "channel", "valence_mass":
            if len(set(subset[field])) > 1:
                raise ValueError(f"Multiple values for {field} returned.")

        return (
            subset[["stream_name", "cfg_index", "correlator"]]
            .set_index(["stream_name", "cfg_index"])
            .correlator
        )

    def get_array(self, **criteria):
        """
        Get the bare correlators for a specific channel as a Numpy array.

        Arguments:
            **criteria: Fields to filter on, and the filter values.
        """
        return np.asarray(self.get(**criteria).to_list())

    @property
    def NT(self):
        if self._frozen:
            return len(self.correlators.correlator.iloc[0])
        else:
            return len(self.correlators[0].correlator)

    @property
    def is_consistent(self):
        """
        Check that the metadata and data are internally consistent.
        """

        if self.metadata.get("NT") != self.NT:
            logging.info(
                "Number of correlator elements doesn't match lattice temporal size!"
            )
            return False

        if self._frozen:
            counts = self.correlators.value_counts(
                subset=["source_type", "connection_type", "channel", "valence_mass"]
            )
            if max(counts) != min(counts):
                logging.info(
                    "Different numbers of observations for different channels. "
                    "Statistics may be funky."
                )
                return False

        return True

    @property
    def frozen(self):
        return self._frozen

    def get_pyerrors(
        self, symmetry=None, enforce_positive_start=True, tag=None, **criteria
    ):
        channel_subset = self.get(**criteria).reset_index()

        NT = self.NT
        all_samples = [[] for _ in range(NT)]
        ensemble_names = sorted(set(channel_subset.stream_name))
        sample_idxs = []

        for ensemble_name in ensemble_names:
            ensemble_subset = channel_subset[
                channel_subset.stream_name == ensemble_name
            ]
            for samples in all_samples:
                samples.append([])

            sample_idxs.append([])
            for correlator in ensemble_subset.itertuples():
                sample_idxs[-1].append(correlator.cfg_index)
                for samples, corr_value in zip(all_samples, correlator.correlator):
                    samples[-1].append(corr_value)

        observables = [
            pe.Obs(samples, ensemble_names, idl=sample_idxs) for samples in all_samples
        ]

        corr = pe.Corr(observables)
        corr.tag = tag

        if symmetry == "antisymmetric":
            corr = corr.anti_symmetric()
        elif symmetry == "symmetric":
            corr = corr.symmetric()

        if enforce_positive_start and corr[1] < 0:
            corr = -corr

        return corr


Correlator = namedtuple(
    "Correlator",
    [
        "stream_name",
        "cfg_index",
        "source_type",
        "connection_type",
        "channel",
        "valence_mass",
        "correlator",
    ],
)
