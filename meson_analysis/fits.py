#!/usr/bin/env python3

import logging

from autograd.numpy import sinh
from pyerrors.fits import least_squares

from .fit_forms import get_fit_form, flat_fit_form
from .readers import read_correlators_hirep


def fit_single_correlator(correlator_ensemble, channel, plateau_range):
    correlator = correlator_ensemble.get_pyerrors(channel=channel)
    correlator.set_prange(plateau_range)
    correlator.gamma_method()

    fit_form = get_fit_form(correlator_ensemble.NT, "v")
    result = correlator.fit(fit_form, silent=True)
    result.gamma_method()
    return result.fit_parameters


def get_multi_correlators(correlator_ensemble, channels, **filters):
    correlators = {
        channel: correlator_ensemble.get_pyerrors(channel=channel, **filters)
        for channel in channels
    }
    for correlator in correlators.values():
        correlator.gamma_method()

    return correlators


def mean_multi_eff_mass(correlator_ensemble, channels, parities={}, **filters):
    correlators = get_multi_correlators(correlator_ensemble, channels, **filters)
    eff_masses = []

    for channel, correlator in correlators.items():
        variant = {None: "log", +1: "cosh", -1: "sinh"}.get(parities.get(channel))
        eff_masses.append(correlator.m_eff(variant=variant))
        eff_masses[-1].gamma_method()

    eff_mass = sum(eff_masses) / len(eff_masses)
    eff_mass.gamma_method()
    return eff_mass


def combine_multi_correlators(correlator_ensemble, channels, parity=None, **filters):
    correlators = get_multi_correlators(correlator_ensemble, channels, **filters)
    combined_correlator = sum(correlators.values())
    if parity == +1:
        combined_correlator = combined_correlator.symmetric()
    elif parity == -1:
        combined_correlator = combined_correlator.anti_symmetric()
    combined_correlator.gamma_method()
    return combined_correlator


def fit_multi_correlators(
    correlator_ensemble, channels, plateau_range, fit_forms=None, full=False, **filters
):
    correlators = get_multi_correlators(correlator_ensemble, channels, **filters)
    x = {
        channel: list(range(plateau_range[0], plateau_range[1] + 1))
        for channel in channels
    }
    y = {
        channel: [v[0] for v in correlator[plateau_range[0] : plateau_range[1] + 1]]
        for channel, correlator in correlators.items()
    }
    if fit_forms is None:
        fit_forms = {
            channel: get_fit_form(correlator_ensemble.NT, "v") for channel in channels
        }
    else:
        assert set(fit_forms.keys()) == set(channels)

    result = least_squares(x, y, fit_forms, silent=True)
    result.gamma_method()

    if full:
        return result
    else:
        return result.fit_parameters


def pcac_eff_mass(correlator_ensemble, **filters):
    g5 = correlator_ensemble.get_pyerrors(channel="g5", **filters)
    g5_g0g5_re = correlator_ensemble.get_pyerrors(channel="g5_g0g5_re", **filters)

    g5.gamma_method()
    g5_eff_mass = g5.m_eff(variant="cosh")

    correction = g5_eff_mass / g5_eff_mass.sinh()

    g5_g0g5_re.gamma_method()
    pcac_eff_mass = correction * g5_g0g5_re.deriv() / (2 * g5)

    if pcac_eff_mass[pcac_eff_mass.T // 2] < 0:
        # Ensure masses are positive
        pcac_eff_mass = -pcac_eff_mass

    pcac_eff_mass.gamma_method()
    return pcac_eff_mass


def fit_pcac(correlator_ensemble, plateau_range, filters={}, full=False):
    eff_mass = pcac_eff_mass(correlator_ensemble, **filters)
    result = eff_mass.fit(flat_fit_form, fitrange=plateau_range, silent=True)
    result.gamma_method()

    if full:
        return result
    else:
        return result[0]


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
