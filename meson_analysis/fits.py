#!/usr/bin/env python3

import logging

from autograd.numpy import sinh
from pyerrors.fits import least_squares

from .fit_forms import get_fit_form
from .readers import read_correlators_hirep


def fit_single_correlator(correlator_ensemble, channel, plateau_range):
    correlator = correlator_ensemble.get_pyerrors(channel=channel)
    correlator.set_prange(plateau_range)
    correlator.gamma_method()

    fit_form = get_fit_form(correlator_ensemble.NT, "v")
    result = correlator.fit(fit_form)
    result.gamma_method()
    return result.fit_parameters


def fit_multi_correlator(correlator_ensemble, channels, plateau_range):
    correlators = {
        channel: correlator_ensemble.get_pyerrors(channel=channel)
        for channel in channels
    }
    for correlator in correlators.values():
        correlator.gamma_method()

    x = {
        channel: list(range(plateau_range[0], plateau_range[1] + 1))
        for channel in channels
    }
    y = {
        channel: [v[0] for v in correlator[plateau_range[0] : plateau_range[1] + 1]]
        for channel, correlator in correlators.items()
    }
    fit_forms = {
        channel: get_fit_form(correlator_ensemble.NT, "v") for channel in channels
    }

    result = least_squares(x, y, fit_forms)
    result.gamma_method()
    return result.fit_parameters


def fit_pcac(correlator_ensemble, plateau_range):
    g5 = correlator_ensemble.get_pyerrors(channel="g5")
    g5_g0g5_re = correlator_ensemble.get_pyerrors(channel="g5_g0g5_re")

    g5.gamma_method()
    g5_eff_mass = g5.m_eff(variant="cosh")

    correction = g5_eff_mass / g5_eff_mass.sinh()

    g5_g0g5_re.gamma_method()
    pcac_eff_mass = correction * g5_g0g5_re.deriv() / (2 * g5)
    pcac_eff_mass.gamma_method()

    result = pcac_eff_mass.plateau(plateau_range=plateau_range)
    result.gamma_method()
    return result


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("plateau_start", type=int)
    parser.add_argument("plateau_end", type=int)
    args = parser.parse_args()

    correlator_ensemble = read_correlators_hirep(args.filename)
    g5_mass, g5_decay_const = fit_single_correlator(
        correlator_ensemble, "g5", [args.plateau_start, args.plateau_end]
    )
    print(f"g5 mass is {g5_mass}; decay const is {g5_decay_const}")

    pcac_mass = fit_pcac(correlator_ensemble, [args.plateau_start, args.plateau_end])
    print(f"PCAC mass is {pcac_mass}")

    gk_mass, gk_decay_const = fit_multi_correlator(
        correlator_ensemble, ["g1", "g2", "g3"], [args.plateau_start, args.plateau_end]
    )
    print(f"gk mass is {gk_mass}; decay const is {gk_decay_const}")


if __name__ == "__main__":
    main()
