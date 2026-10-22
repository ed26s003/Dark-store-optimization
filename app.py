import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from optimizer import (
    generate_instance, haversine_matrix, solve_location_model,
    build_kpis, build_store_results, build_customer_results,
    run_profit_experiment, validate_solution
)

st.set_page_config(page_title="Dark Store Optimizer", page_icon="🏬", layout="wide")
st.title("🏬 Dark Store Location Optimization")
st.caption("MILP decision support • Chennai, Tamil Nadu • Synthetic data unless otherwise stated")

with st.sidebar:
    st.header("Problem inputs")
    n_stores = st.number_input("Candidate dark store locations", min_value=1, max_value=100, value=20, step=1)
    n_customers = st.number_input("Number of customers", min_value=1, max_value=10000, value=250, step=25)
    max_distance = st.number_input("Maximum customer-to-store distance (km)", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
    max_open = st.number_input("Maximum stores allowed to open (K)", min_value=1, max_value=int(n_stores), value=min(5, int(n_stores)), step=1)
    st.subheader("Demand and economics")
    demand_mode = st.selectbox("Daily demand generation", ["Random integer demand", "Same demand for every customer"])
    if demand_mode == "Random integer demand":
        demand_low = st.number_input("Minimum orders per customer/day", min_value=1, max_value=100, value=1)
        demand_high = st.number_input("Maximum orders per customer/day", min_value=int(demand_low), max_value=200, value=5)
    else:
        fixed_demand = st.number_input("Orders per customer/day", min_value=1, max_value=200, value=2)
        demand_low, demand_high = int(fixed_demand), int(fixed_demand)
    revenue = st.number_input("Revenue / contribution per order (₹)", min_value=0.0, value=120.0, step=5.0)
    delivery_cost = st.number_input("Delivery cost (₹ per order per km)", min_value=0.0, value=8.0, step=0.5)
    fixed_cost = st.number_input("Fixed operating cost per store/day (₹)", min_value=0.0, value=12000.0, step=500.0)
    capacity = st.number_input("Daily order capacity per store", min_value=1, max_value=100000, value=1500, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42, step=1)
    time_limit = st.number_input("Solver time limit (seconds)", min_value=1, max_value=600, value=30, step=5)
    mip_gap = st.number_input("Relative MIP gap target (0 = exact proof)", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.2f")
    run = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

st.info(
    "The model maximizes estimated daily profit. Each customer is either assigned to at most one open store or left unserved. "
    "Unserved demand generates neither revenue nor delivery cost. The distance cap is enforced inside the MILP; it is not a map-only filter. "
    "All locations and demands are reproducible synthetic data, not actual customers or businesses."
)

with st.expander("Model formulation and assumptions"):
    st.markdown(r"""
**Decision variables:** \(x_{ij}\in\{0,1\}\) indicates customer \(i\) is assigned to store \(j\); \(y_j\in\{0,1\}\) indicates store \(j\) opens.

**Maximize daily profit**
\[
\sum_{(i,j):D_{ij}\le D_{\max}} d_i(r-cD_{ij})x_{ij}-\sum_jF_jy_j
\]
subject to:
- **Assignment:** \(\sum_jx_{ij}\le1\) for each customer.
- **Open-store linkage:** \(x_{ij}\le y_j\) for each eligible pair.
- **Capacity:** \(\sum_i d_ix_{ij}\le C_jy_j\) for each store.
- **Hard distance:** pairs with \(D_{ij}>D_{\max}\) are omitted from the model, so they cannot be selected.
- **Store limit:** \(\sum_j y_j\le K\).
- **Domains:** \(x_{ij},y_j\) are binary.

Distances use Haversine great-circle distance (straight line), not road distance or travel time. The lecture PDF (pp. 26–28) uses a demand-share variable; this app intentionally uses the assignment's binary assignment formulation.
""")

if run:
    if max_open > n_stores:
        st.error("K cannot exceed the number of candidate locations.")
        st.stop()
    if demand_high < demand_low:
        st.error("Maximum demand must be greater than or equal to minimum demand.")
        st.stop()
    with st.spinner("Generating data, calculating distances, and solving the MILP..."):
        customers, stores = generate_instance(
            int(n_customers), int(n_stores), int(seed),
            int(demand_low), int(demand_high)
        )
        distances = haversine_matrix(customers, stores)
        started = time.perf_counter()
        result = solve_location_model(
            customers, stores, distances, float(max_distance), int(max_open),
            float(revenue), float(delivery_cost), float(fixed_cost), int(capacity),
            time_limit=int(time_limit), mip_rel_gap=float(mip_gap)
        )
        elapsed = time.perf_counter() - started
    if result.get("x_values") is None:
        st.error(f"No feasible incumbent was returned. Solver status: {result['status']}. {result.get('message','')}")
        st.caption(f"Elapsed: {elapsed:.2f} seconds. Try increasing the time limit, increasing K/capacity, or loosening the distance limit.")
        st.stop()

    validation = validate_solution(customers, stores, distances, result, float(max_distance), int(max_open), int(capacity))
    if not validation["valid"]:
        st.error("The returned solution failed validation: " + "; ".join(validation["errors"]))
        st.stop()

    kpis = build_kpis(customers, stores, distances, result, float(max_distance), float(revenue), float(delivery_cost), float(fixed_cost), int(capacity), elapsed)
    store_results = build_store_results(customers, stores, distances, result, float(revenue), float(delivery_cost), int(capacity))
    customer_results = build_customer_results(customers, stores, distances, result)

    st.session_state["last_run"] = {
        "customers": customers, "stores": stores, "distances": distances,
        "result": result, "kpis": kpis, "store_results": store_results,
        "customer_results": customer_results, "max_distance": float(max_distance),
        "capacity": int(capacity), "revenue": float(revenue),
        "delivery_cost": float(delivery_cost), "fixed_cost": float(fixed_cost),
        "K": int(max_open), "elapsed": elapsed
    }

if "last_run" in st.session_state:
    data = st.session_state["last_run"]
    kpis = data["kpis"]
    st.subheader("Optimization results")
    status = kpis.loc[kpis["Metric"] == "Solver status", "Value"].iloc[0]
    if "optimal" in str(status).lower():
        st.success(f"Solver status: {status}")
    else:
        st.warning(f"Solver status: {status}. Treat this as feasible, not proven optimal, unless the status explicitly says optimal.")
    cols = st.columns(4)
    metric_lookup = dict(zip(kpis["Metric"], kpis["Value"]))
    cols[0].metric("Estimated daily profit", f"₹{float(metric_lookup['Estimated daily profit']):,.2f}")
    cols[1].metric("Stores opened", f"{int(metric_lookup['Stores opened'])} / {data['K']}")
    cols[2].metric("Customers served", f"{int(metric_lookup['Customers served']):,}")
    cols[3].metric("Demand served", f"{float(metric_lookup['Demand served percentage']):.1f}%")
    st.dataframe(kpis, use_container_width=True, hide_index=True)
    st.download_button("Download KPI CSV", kpis.to_csv(index=False).encode("utf-8"), "dark_store_kpis.csv", "text/csv")
    st.subheader("Interactive Chennai solution map")
    st.caption("Green star = open store; grey square = candidate store not opened; blue circle = served customer; red x = unserved customer. Lines are assignment links.")
    customers = data["customers"].copy()
    stores = data["stores"].copy()
    assignments = data["customer_results"]
    opened = set(data["result"]["opened_stores"])
    fig = go.Figure()
    unopened = stores[~stores["store_id"].isin(opened)]
    if not unopened.empty:
        fig.add_trace(go.Scattermapbox(
            lat=unopened.latitude, lon=unopened.longitude, mode="markers",
            marker=dict(size=10, color="gray", symbol="square"),
            name="Candidate store (closed)",
            text=unopened.store_id, hovertemplate="Closed %{text}<extra></extra>"
        ))
    opened_df = stores[stores["store_id"].isin(opened)]
    if not opened_df.empty:
        fig.add_trace(go.Scattermapbox(
            lat=opened_df.latitude, lon=opened_df.longitude, mode="markers",
            marker=dict(size=16, color="green", symbol="star"),
            name="Open store", text=opened_df.store_id,
            hovertemplate="Open %{text}<br>Lat %{lat:.5f}<br>Lon %{lon:.5f}<extra></extra>"
        ))
    served = assignments[assignments["status"] == "Served"]
    unserved = assignments[assignments["status"] != "Served"]
    if not served.empty:
        fig.add_trace(go.Scattermapbox(
            lat=served.latitude, lon=served.longitude, mode="markers",
            marker=dict(size=7, color="#168aad"),
            name="Served customer",
            text=served.apply(lambda r: f"{r.customer_id}<br>Store: {r.assigned_store}<br>Demand: {r.daily_demand}<br>Distance: {r.distance_km:.2f} km", axis=1),
            hovertemplate="%{text}<extra></extra>"
        ))
    if not unserved.empty:
        fig.add_trace(go.Scattermapbox(
            lat=unserved.latitude, lon=unserved.longitude, mode="markers",
            marker=dict(size=7, color="red", symbol="x"),
            name="Unserved customer", text=unserved.customer_id,
            hovertemplate="%{text}<br>Unserved under distance/capacity/economic constraints<extra></extra>"
        ))
    show_lines = st.checkbox("Show assignment lines (may be slow for large instances)", value=len(served) <= 1500)
    line_store_filter = st.selectbox("Assignment-line filter", ["All open stores"] + sorted(opened))
    if show_lines:
        line_df = served
        if line_store_filter != "All open stores":
            line_df = served[served.assigned_store == line_store_filter]
        if len(line_df) > 2500:
            st.warning("Showing only the first 2,500 assignment lines for responsiveness. Customer/store assignments in the table remain complete.")
            line_df = line_df.head(2500)
        for _, row in line_df.iterrows():
            store_row = stores.loc[stores.store_id == row.assigned_store].iloc[0]
            fig.add_trace(go.Scattermapbox(
                lat=[row.latitude, store_row.latitude], lon=[row.longitude, store_row.longitude],
                mode="lines", line=dict(width=1, color="rgba(50,100,130,0.35)"),
                showlegend=False, hoverinfo="skip"
            ))
    center_lat = float(stores.latitude.mean()) if len(stores) else 13.08
    center_lon = float(stores.longitude.mean()) if len(stores) else 80.27
    fig.update_layout(mapbox_style="open-street-map", mapbox_center={"lat": center_lat, "lon": center_lon},
                      mapbox_zoom=9, height=650, margin={"l":0,"r":0,"t":10,"b":0},
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Store-level results")
    st.dataframe(data["store_results"], use_container_width=True, hide_index=True)
    st.download_button("Download store results CSV", data["store_results"].to_csv(index=False).encode("utf-8"), "store_results.csv", "text/csv")
    st.subheader("Customer assignment details")
    st.dataframe(data["customer_results"], use_container_width=True, hide_index=True)
    st.download_button("Download customer assignments CSV", data["customer_results"].to_csv(index=False).encode("utf-8"), "customer_assignments.csv", "text/csv")

st.divider()
st.subheader("Required profit experiment: 10,000 customers, 50 candidate stores")
st.write("Solves five separate models for K = 2, 5, 10, 20, 25 using the same synthetic customers, candidate stores, distances, and economics.")
with st.expander("Experiment configuration"):
    exp_distance = st.number_input("Experiment maximum distance (km)", min_value=0.1, max_value=50.0, value=2.0, step=0.1, key="exp_dist")
    exp_capacity = st.number_input("Experiment capacity per store/day", min_value=1, max_value=100000, value=1500, step=100, key="exp_cap")
    exp_revenue = st.number_input("Experiment revenue per order (₹)", min_value=0.0, value=120.0, step=5.0, key="exp_rev")
    exp_delivery = st.number_input("Experiment delivery cost (₹/order/km)", min_value=0.0, value=8.0, step=0.5, key="exp_del")
    exp_fixed = st.number_input("Experiment fixed cost/store/day (₹)", min_value=0.0, value=12000.0, step=500.0, key="exp_fix")
    exp_limit = st.number_input("Time limit per experiment solve (seconds)", min_value=1, max_value=600, value=20, step=5, key="exp_time")
    exp_seed = st.number_input("Experiment seed", min_value=0, max_value=2**31-1, value=42, step=1, key="exp_seed")
if st.button("Run five-store-limit profit experiment", type="secondary"):
    with st.spinner("Running five 10,000-customer MILP instances. This can take several minutes..."):
        exp_df = run_profit_experiment(
            max_distance=float(exp_distance), capacity=int(exp_capacity),
            revenue=float(exp_revenue), delivery_cost=float(exp_delivery),
            fixed_cost=float(exp_fixed), seed=int(exp_seed),
            time_limit=int(exp_limit), mip_rel_gap=float(mip_gap)
        )
    st.session_state["experiment_results"] = exp_df
if "experiment_results" in st.session_state:
    exp_df = st.session_state["experiment_results"]
    st.dataframe(exp_df, use_container_width=True, hide_index=True)
    st.download_button("Download experiment CSV", exp_df.to_csv(index=False).encode("utf-8"), "profit_experiment.csv", "text/csv")
    chart_df = exp_df[pd.to_numeric(exp_df["Estimated daily profit (₹)"], errors="coerce").notna()]
    if not chart_df.empty:
        chart = px.line(chart_df, x="Maximum stores allowed (K)", y="Estimated daily profit (₹)", markers=True,
                        title="Estimated daily profit vs. maximum allowed stores")
        st.plotly_chart(chart, use_container_width=True)
    st.caption("For the true optimum, profit cannot decrease as K increases because every solution feasible for a smaller K remains feasible for a larger K. Time-limited incumbent solutions may fluctuate and are not proof of that monotonicity.")
