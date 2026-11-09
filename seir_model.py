
import numpy as np
from rk4_solver import rk4_step


def seir_derivatives(state, N, beta, sigma, gamma):
    S, E, I, R = state

    dS = -beta * S * I / N
    dE = (beta * S * I / N) - (sigma * E)
    dI = (sigma * E) - (gamma * I)
    dR = gamma * I

    return (dS, dE, dI, dR)


def seir_model(N, S0, E0, I0, R0,
               beta, sigma, gamma, days, dt=0.1):

    if N <= 0 or days <= 0 or dt <= 0:
        raise ValueError("Population, days and dt must be positive.")

    if min(S0, E0, I0, R0) < 0:
        raise ValueError("Initial populations cannot be negative.")

    if not np.isclose(S0 + E0 + I0 + R0, N):
        raise ValueError("Initial populations must add up to N.")

    if min(beta, sigma, gamma) < 0:
        raise ValueError("Model rates cannot be negative.")

    steps = int(round(days / dt))
    t = np.linspace(0, days, steps + 1)

    results = np.zeros((steps + 1, 4))
    results[0] = [S0, E0, I0, R0]

    def derivative_function(state):
        return seir_derivatives(
            state, N, beta, sigma, gamma
        )

    for j in range(steps):
        results[j + 1] = rk4_step(
            derivative_function,
            tuple(results[j]),
            dt
        )

    S = results[:, 0]
    E = results[:, 1]
    I = results[:, 2]
    R = results[:, 3]

    return t, S, E, I, R
