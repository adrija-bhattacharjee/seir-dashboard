
import numpy as np
import csv
from seir_model import seir_model
from hospital_forecast import hospital_forecast

# Population and initial conditions
N = 100000
S0 = 99900
E0 = 50
I0 = 50
R0 = 0

days = 160
dt = 0.1

# Different simulation scenarios
scenarios = [
    {
        "name": "Low Transmission",
        "beta": 0.20,
        "sigma": 0.20,
        "gamma": 0.10
    },
    {
        "name": "Normal Transmission",
        "beta": 0.30,
        "sigma": 0.20,
        "gamma": 0.10
    },
    {
        "name": "High Transmission",
        "beta": 0.45,
        "sigma": 0.20,
        "gamma": 0.10
    },
    {
        "name": "Faster Recovery",
        "beta": 0.30,
        "sigma": 0.20,
        "gamma": 0.15
    }
]

results = []

print("\n===== SEIR SCENARIO TESTING =====")

for scenario in scenarios:

    t, S, E, I, R = seir_model(
        N, S0, E0, I0, R0,
        scenario["beta"],
        scenario["sigma"],
        scenario["gamma"],
        days, dt
    )

    hospital, icu = hospital_forecast(I)

    peak_index = np.argmax(I)

    peak_infections = int(np.max(I))
    peak_day = round(t[peak_index], 1)
    peak_hospital_beds = int(np.max(hospital))
    peak_icu_beds = int(np.max(icu))

    results.append([
        scenario["name"],
        scenario["beta"],
        scenario["gamma"],
        peak_infections,
        peak_day,
        peak_hospital_beds,
        peak_icu_beds
    ])

    print("\nScenario:", scenario["name"])
    print("Peak infections:", peak_infections)
    print("Peak infection day:", peak_day)
    print("Peak hospital beds:", peak_hospital_beds)
    print("Peak ICU beds:", peak_icu_beds)

# Save scenario comparison
with open("scenario_results.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Scenario",
        "Beta",
        "Gamma",
        "Peak Infections",
        "Peak Day",
        "Hospital Beds",
        "ICU Beds"
    ])

    writer.writerows(results)

print("\nScenario testing completed!")
print("Results saved in scenario_results.csv")
