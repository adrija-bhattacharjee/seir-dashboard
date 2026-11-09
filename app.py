"""
SEIR Epidemic Simulator - Streamlit Dashboard (Member 3)

Run with:   streamlit run app.py
Needs (same folder): seir_model.py, rk4_solver.py, hospital_forecast.py
"""

import numpy as np
import csv
import io
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seir_model import seir_model
from hospital_forecast import hospital_forecast, get_peak_bed_demand

# ------------------------------------------------------------------ page setup
st.set_page_config(
    page_title="SEIR Epidemic Simulator",
    page_icon="🦠",
    layout="wide",
)

COLORS = {
    "S": "#2E86DE",   # blue
    "E": "#F39C12",   # orange
    "I": "#E74C3C",   # red
    "R": "#27AE60",   # green
    "H": "#8E44AD",   # purple
    "ICU": "#C0392B", # dark red
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.10);
        border: 1px solid rgba(128,128,128,0.25);
        padding: 14px 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def run_simulation(N, E0, I0, beta, sigma, gamma, days, dt, hosp_rate, icu_rate):
    """Runs Member 2's RK4 SEIR model + hospital forecast."""
    S0 = N - E0 - I0
    t, S, E, I, R = seir_model(N, S0, E0, I0, 0, beta, sigma, gamma, days, dt)
    beds, icu = hospital_forecast(I, hosp_rate, icu_rate)
    return t, S, E, I, R, beds, icu


def style_axes(ax, title, ylabel="Number of People"):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


def seir_figure(t, S, E, I, R, peak_day):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, S, label="Susceptible", color=COLORS["S"], lw=2)
    ax.plot(t, E, label="Exposed", color=COLORS["E"], lw=2)
    ax.plot(t, I, label="Infected", color=COLORS["I"], lw=2.5)
    ax.plot(t, R, label="Recovered", color=COLORS["R"], lw=2)
    ax.axvline(peak_day, color=COLORS["I"], ls="--", alpha=0.6)
    ax.annotate(f"Peak: day {peak_day:.0f}", (peak_day, I.max()),
                xytext=(8, 8), textcoords="offset points", color=COLORS["I"])
    style_axes(ax, "SEIR Epidemic Model (RK4)")
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def hospital_figure(t, beds, icu, capacity=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, beds, label="Hospital beds needed", color=COLORS["H"], lw=2.5)
    ax.plot(t, icu, label="ICU beds needed", color=COLORS["ICU"], lw=2.5)
    ax.fill_between(t, beds, alpha=0.12, color=COLORS["H"])
    if capacity:
        ax.axhline(capacity, color="black", ls="--", lw=1.5,
                   label=f"Available capacity ({capacity:,})")
    style_axes(ax, "Hospital Bed Demand Forecast", "Beds Required")
    ax.legend()
    fig.tight_layout()
    return fig


def md_table(rows):
    """Shows a list of dicts as a table without needing pandas."""
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = [f"{r[c]:,}" if isinstance(r[c], int) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ header
st.title("🦠 SEIR Epidemic Simulator")
st.caption(
    "Interactive dashboard: change the settings on the left and the "
    "simulation (Runge-Kutta 4 SEIR model) re-runs instantly."
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Simulation Settings")

    st.subheader("Population")
    N = st.number_input("Total population", min_value=1000, max_value=100_000_000,
                        value=100_000, step=10_000)
    I0 = st.slider("Initially infected people", 1, 5000, 50)
    E0 = st.slider("Initially exposed people", 0, 5000, 50,
                   help="People infected but not yet infectious.")

    st.subheader("Disease")
    beta = st.slider("Infection rate (β)", 0.05, 1.50, 0.30, 0.01,
                     help="How fast the disease spreads per day.")
    gamma = st.slider("Recovery rate (γ)", 0.02, 0.50, 0.10, 0.01,
                      help="1/γ = average days a person stays infectious.")
    sigma = st.slider("Incubation rate (σ)", 0.05, 1.00, 0.20, 0.01,
                      help="1/σ = average days before an exposed person becomes infectious.")

    st.subheader("Duration")
    days = st.slider("Simulation duration (days)", 30, 730, 160, 10)

    st.subheader("Hospital")
    hosp_rate = st.slider("Hospitalization rate (% of infected)", 1, 30, 5) / 100
    icu_rate = st.slider("ICU rate (% of hospitalized)", 1, 60, 20) / 100
    capacity = st.number_input("Available hospital beds (optional, 0 = hide)",
                               min_value=0, value=0, step=100)

    with st.expander("Advanced"):
        dt = st.select_slider("Time step (days)", options=[0.5, 0.25, 0.1, 0.05], value=0.1,
                              help="Smaller = more accurate but slower.")

    if st.button("↺ Reset to defaults", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ------------------------------------------------------------------ validation
if I0 + E0 >= N:
    st.error("Initially infected + exposed people must be less than the total population.")
    st.stop()

# ------------------------------------------------------------------ run model
try:
    t, S, E, I, R, beds, icu = run_simulation(
        N, E0, I0, beta, sigma, gamma, days, dt, hosp_rate, icu_rate
    )
except ValueError as err:
    st.error(f"Simulation error: {err}")
    st.stop()

peak_idx = int(np.argmax(I))
peak_inf = int(I[peak_idx])
peak_day = float(t[peak_idx])
peak_beds = get_peak_bed_demand(beds, icu)
R0_basic = beta / gamma
total_infected = N - S[-1]

# ------------------------------------------------------------------ key numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak active infections", f"{peak_inf:,}", f"{peak_inf / N:.1%} of population",
          delta_color="off")
c2.metric("Day of peak", f"Day {peak_day:.0f}")
c3.metric("Peak hospital beds", f"{peak_beds['maximum_hospital_beds']:,}")
c4.metric("Peak ICU beds", f"{peak_beds['maximum_icu_beds']:,}")

c5, c6, c7 = st.columns(3)
c5.metric("Basic reproduction number (R₀ = β/γ)", f"{R0_basic:.2f}")
c6.metric("Total ever infected", f"{total_infected:,.0f}", f"{total_infected / N:.1%} of population",
          delta_color="off")
c7.metric("Still susceptible at end", f"{S[-1]:,.0f}")

if R0_basic < 1:
    st.success("R₀ < 1: the outbreak dies out on its own.")
else:
    st.warning(f"R₀ = {R0_basic:.2f} > 1: the epidemic spreads through the population.")

if capacity and peak_beds["maximum_hospital_beds"] > capacity:
    over = peak_beds["maximum_hospital_beds"] - capacity
    days_over = int(np.sum(beds > capacity) * dt)
    st.error(f"🚨 Hospitals overloaded: peak demand exceeds capacity by {over:,} beds "
             f"(about {days_over} days over capacity).")
elif capacity:
    st.success("✅ Hospital capacity is enough for the whole simulation.")

# ------------------------------------------------------------------ tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Epidemic Curves", "🏥 Hospital Demand", "🔀 Scenario Comparison", "📄 Data & Download"]
)

with tab1:
    show_figure(seir_figure(t, S, E, I, R, peak_day))

with tab2:
    show_figure(hospital_figure(t, beds, icu, capacity or None))
    st.caption(f"Assumes {hosp_rate:.0%} of infected people need a hospital bed and "
               f"{icu_rate:.0%} of those need an ICU bed.")

with tab3:
    st.write("Compare transmission scenarios using your current population, "
             "duration and hospital settings.")
    scenarios = {
        "Low Transmission": (0.20, 0.10),
        "Normal Transmission": (0.30, 0.10),
        "High Transmission": (0.45, 0.10),
        "Faster Recovery": (0.30, 0.15),
        "Your current settings": (beta, gamma),
    }
    rows = []
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, (b, g) in scenarios.items():
        ts, _, _, Is, _, bs, ics = run_simulation(
            N, E0, I0, b, sigma, g, days, dt, hosp_rate, icu_rate)
        k = int(np.argmax(Is))
        rows.append({
            "Scenario": name, "β": b, "γ": g, "R₀": round(b / g, 2),
            "Peak infections": int(Is[k]), "Peak day": round(float(ts[k]), 1),
            "Hospital beds": int(bs.max()), "ICU beds": int(ics.max()),
        })
        ax.plot(ts, Is, label=name, lw=3 if name.startswith("Your") else 1.8,
                ls="--" if name.startswith("Your") else "-")
    style_axes(ax, "Active Infections by Scenario")
    ax.legend()
    fig.tight_layout()
    show_figure(fig)
    md_table(rows)

with tab4:
    header = ["Day", "Susceptible", "Exposed", "Infected", "Recovered",
              "Hospital_Beds", "ICU_Beds"]
    all_rows = [
        [round(float(t[i]), 2), round(float(S[i]), 2), round(float(E[i]), 2),
         round(float(I[i]), 2), round(float(R[i]), 2), int(beds[i]), int(icu[i])]
        for i in range(len(t))
    ]
    # Table: show every 10th day to keep it readable
    step = max(1, int(round(10 / dt)))
    table_rows = [
        {"Day": int(r[0]), "Susceptible": int(r[1]), "Exposed": int(r[2]),
         "Infected": int(r[3]), "Recovered": int(r[4]),
         "Hospital beds": r[5], "ICU beds": r[6]}
        for r in all_rows[::step]
    ]
    st.caption("Showing every 10th day. Download the CSV for the full data.")
    md_table(table_rows)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(all_rows)
    st.download_button("⬇️ Download full results (CSV)", buf.getvalue(),
                       file_name="results.csv", mime="text/csv")

    ok_pop = np.allclose(S + E + I + R, N, atol=1)
    ok_neg = bool(np.all(np.array([S, E, I, R]) >= -1e-6))
    st.markdown("**Model validation**")
    st.write(f"{'✅' if ok_pop else '❌'} Population conserved  \n"
             f"{'✅' if ok_neg else '❌'} No negative populations")

st.divider()
st.caption("SEIR model: dS/dt = −βSI/N, dE/dt = βSI/N − σE, dI/dt = σE − γI, dR/dt = γI")
