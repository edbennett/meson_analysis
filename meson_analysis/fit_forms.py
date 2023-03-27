#!/usr/bin/env python3

from functools import partial

from autograd.numpy import exp


def ps_fit_form(params, t, NT=None):
    mass = params[0]
    decay_const = params[1]
    amplitude = params[2]
    return amplitude**2 / mass * (exp(-mass * t) + exp(-mass * (NT - t)))


def ps_av_fit_form(params, t, NT=None):
    mass = params[0]
    decay_const = params[1]
    amplitude = params[2]
    return amplitude * decay_const * (exp(-mass * t) - exp(-mass * (NT - t)))


def v_fit_form(params, t, NT=None):
    mass = params[0]
    decay_const = params[1]
    return decay_const**2 * mass * (exp(-mass * t) + exp(-mass * (NT - t)))


def get_fit_form(NT, fit_form):
    return partial(
        {"ps": ps_fit_form, "ps_av": ps_av_fit_form, "v": v_fit_form}[fit_form],
        NT=NT,
    )
