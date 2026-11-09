
def rk4_step(derivatives, state, dt):
    """
    Performs one Runge-Kutta 4th order integration step.
    """

    k1 = derivatives(state)

    state2 = tuple(
        state[i] + dt * k1[i] / 2
        for i in range(len(state))
    )
    k2 = derivatives(state2)

    state3 = tuple(
        state[i] + dt * k2[i] / 2
        for i in range(len(state))
    )
    k3 = derivatives(state3)

    state4 = tuple(
        state[i] + dt * k3[i]
        for i in range(len(state))
    )
    k4 = derivatives(state4)

    new_state = tuple(
        state[i] + (dt / 6) * (
            k1[i] + 2*k2[i] + 2*k3[i] + k4[i]
        )
        for i in range(len(state))
    )

    return new_state
