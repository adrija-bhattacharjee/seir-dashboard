
import numpy as np


def hospital_forecast(
    infected,
    hospitalization_rate=0.05,
    icu_rate=0.20
):
    """
    Estimates hospital and ICU bed requirements.

    hospitalization_rate: Fraction of infected people
                           requiring hospital beds.

    icu_rate: Fraction of hospitalized people
              requiring ICU beds.
    """

    infected = np.asarray(infected, dtype=float)

    if np.any(infected < 0):
        raise ValueError("Infected population cannot be negative.")

    if not 0 <= hospitalization_rate <= 1:
        raise ValueError("Invalid hospitalization rate.")

    if not 0 <= icu_rate <= 1:
        raise ValueError("Invalid ICU rate.")

    # General hospital bed requirement
    hospital_beds = np.ceil(
        infected * hospitalization_rate
    )

    # ICU bed requirement
    icu_beds = np.ceil(
        hospital_beds * icu_rate
    )

    return hospital_beds, icu_beds


def get_peak_bed_demand(hospital_beds, icu_beds):

    return {
        "maximum_hospital_beds": int(np.max(hospital_beds)),
        "maximum_icu_beds": int(np.max(icu_beds))
    }
