
import numpy as np
import matplotlib.pyplot as plt
import csv

from seir_model import seir_model
from hospital_forecast import hospital_forecast, get_peak_bed_demand

# 1. Population and simulation settings
N = 100000

S0 = 99900
E0 = 50
I0 = 50
R0 = 0

beta = 0.30
sigma = 0.20
gamma = 0.10

days = 160
dt = 0.1

# 2. Run the SEIR model
t, S, E, I, R = seir_model(
    N, S0, E0, I0, R0,
    beta, sigma, gamma, days, dt
)

# 3. Calculate hospital and ICU bed requirements
hospital_beds, icu_beds = hospital_forecast(I)

# 4. Find peak values
peak_day = int(np.argmax(I))

peak_demand = get_peak_bed_demand(
    hospital_beds, icu_beds
)

print("\n===== SEIR EPIDEMIC SIMULATION =====")
print("Total population:", N)
print("Simulation duration:", days, "days")
print("Peak active infections:", int(np.max(I)))
print("Day of peak infections:", round(t[peak_day], 1))

print("\n===== HOSPITAL BED FORECAST =====")
print("Maximum hospital beds:", peak_demand["maximum_hospital_beds"])
print("Maximum ICU beds:", peak_demand["maximum_icu_beds"])

# 5. Save results to CSV
with open("results.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Day", "Susceptible", "Exposed",
        "Infected", "Recovered",
        "Hospital_Beds", "ICU_Beds"
    ])

    for i in range(len(t)):
        writer.writerow([
            round(t[i], 2),
            round(S[i], 2),
            round(E[i], 2),
            round(I[i], 2),
            round(R[i], 2),
            int(hospital_beds[i]),
            int(icu_beds[i])
        ])

print("\nResults saved successfully in results.csv")

# 6. Plot SEIR graph
plt.figure(figsize=(10, 6))

plt.plot(t, S, label="Susceptible")
plt.plot(t, E, label="Exposed")
plt.plot(t, I, label="Infected")
plt.plot(t, R, label="Recovered")

plt.xlabel("Time (Days)")
plt.ylabel("Number of People")
plt.title("SEIR Epidemic Model using RK4")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("seir_graph.png")
plt.show()

# 7. Plot hospital bed forecast
plt.figure(figsize=(10, 6))

plt.plot(t, hospital_beds, label="Hospital Beds")
plt.plot(t, icu_beds, label="ICU Beds")

plt.xlabel("Time (Days)")
plt.ylabel("Required Beds")
plt.title("Hospital Bed Demand Forecast")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("hospital_forecast.png")
plt.show()
# 8. Validate simulation results
total_population = S + E + I + R

print("\n===== MODEL VALIDATION =====")
print("Population conserved:", np.allclose(total_population, N, atol=1))
print("No negative populations:", np.all(
    np.array([S, E, I, R]) >= -1e-6
))
print("\nSimulation completed successfully!")
