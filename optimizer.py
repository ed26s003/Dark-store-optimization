"""Sparse MILP implementation for dark-store location optimization."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

# Chennai-area synthetic bounding box. Points are simulated, not real addresses.
CHENNAI_BOUNDS = {"lat_min": 12.85, "lat_max": 13.25, "lon_min": 80.05, "lon_max": 80.35}

def generate_instance(n_customers: int, n_stores: int, seed: int = 42,
                      demand_min: int = 1, demand_max: int = 5):
    if n_customers < 1 or n_stores < 1:
        raise ValueError("Customer and candidate-store counts must be positive.")
    if demand_min < 1 or demand_max < demand_min:
        raise ValueError("Demand bounds must satisfy 1 <= minimum <= maximum.")
    rng = np.random.default_rng(seed)
    ```python
    def points(n):
        # Generate synthetic points around inland Chennai urban areas.
        # This avoids sampling uniformly across the sea and land together.
        # Coordinates represent approximate area centres, not real addresses.
        land_centres = np.array([
            [12.9249, 80.1000],  # Tambaram
            [13.0382, 80.1565],  # Porur
            [13.0067, 80.2206],  # Guindy
            [13.0850, 80.2101],  # Anna Nagar
            [13.1143, 80.1548],  # Ambattur
            [13.0835, 80.1750],  # Mogappair
            [12.9750, 80.2200],  # Velachery
            [12.9516, 80.1462],  # Chromepet
            [13.0480, 80.0940],  # Poonamallee
            [13.0694, 80.1948],  # Koyambedu
        ])

        # Randomly select an inland area for each point.
        selected = rng.integers(0, len(land_centres), size=n)

        # Add small random variations so points do not overlap exactly.
        # The spread is approximately a few hundred metres.
        lat = land_centres[selected, 0] + rng.normal(0, 0.004, n)
        lon = land_centres[selected, 1] + rng.normal(0, 0.004, n)

        return lat, lon
```
    clat, clon = points(n_customers)
    slat, slon = points(n_stores)
    customers = pd.DataFrame({"customer_id": [f"C{i+1:05d}" for i in range(n_customers)],
                              "latitude": clat, "longitude": clon,
                              "daily_demand": rng.integers(demand_min, demand_max + 1, n_customers)})
    stores = pd.DataFrame({"store_id": [f"S{j+1:03d}" for j in range(n_stores)],
                           "latitude": slat, "longitude": slon})
    return customers, stores

def haversine_matrix(customers: pd.DataFrame, stores: pd.DataFrame) -> np.ndarray:
    """Great-circle distances in kilometres, matrix shape (customers, stores)."""
    lat1 = np.radians(customers["latitude"].to_numpy(dtype=float))[:, None]
    lon1 = np.radians(customers["longitude"].to_numpy(dtype=float))[:, None]
    lat2 = np.radians(stores["latitude"].to_numpy(dtype=float))[None, :]
    lon2 = np.radians(stores["longitude"].to_numpy(dtype=float))[None, :]
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def solve_location_model(customers, stores, distances, max_distance, max_open,
                         revenue_per_order=120.0, delivery_cost_per_order_km=8.0,
                         fixed_cost_per_store=12000.0, capacity_per_store=1500,
                         time_limit=30, mip_rel_gap=0.02):
    """Solve sparse binary assignment MILP with scipy.optimize.milp / HiGHS.

    Returns a result dict. Status 0 means optimal; status 1 means limit reached,
    and a returned incumbent is feasible but not proven optimal.
    """
    n, m = distances.shape
    if n != len(customers) or m != len(stores):
        raise ValueError("Distance matrix dimensions do not match customers/stores.")
    if not (1 <= max_open <= m):
        raise ValueError("max_open must be between 1 and the candidate store count.")
    if max_distance < 0 or capacity_per_store < 1 or time_limit <= 0:
        raise ValueError("Distance, capacity, and time limit must be valid.")
    demands = customers["daily_demand"].to_numpy(dtype=float)
    # Eligible pairs only: forbidden pairs have no x variable at all.
    ii, jj = np.where(distances <= max_distance)
    p = len(ii)
    nvars = p + m
    contribution = demands[ii] * (float(revenue_per_order) - float(delivery_cost_per_order_km) * distances[ii, jj])
    c = np.concatenate([-contribution, np.full(m, float(fixed_cost_per_store))])
    integrality = np.ones(nvars, dtype=np.uint8)
    bounds = Bounds(np.zeros(nvars), np.ones(nvars))

    # Rows: customer assignment (n), store capacity (m), pair linkage (p), store limit (1).
    rows, cols, vals = [], [], []
    lower = np.full(n + m + p + 1, -np.inf)
    upper = np.full(n + m + p + 1, np.inf)
    # sum_j x_ij <= 1
    for k, i in enumerate(ii):
        rows.append(int(i)); cols.append(k); vals.append(1.0)
    upper[:n] = 1.0
    # sum_i demand_i*x_ij - capacity*y_j <= 0
    for k, (i, j) in enumerate(zip(ii, jj)):
        rows.append(n + int(j)); cols.append(k); vals.append(float(demands[i]))
    for j in range(m):
        rows.append(n + j); cols.append(p + j); vals.append(-float(capacity_per_store))
    upper[n:n+m] = 0.0
    # x_ij - y_j <= 0
    for k, j in enumerate(jj):
        rows.extend([n+m+k, n+m+k])
        cols.extend([k, p+int(j)])
        vals.extend([1.0, -1.0])
    upper[n+m:n+m+p] = 0.0
    # sum_j y_j <= K
    for j in range(m):
        rows.append(n+m+p); cols.append(p+j); vals.append(1.0)
    upper[-1] = float(max_open)
    A = coo_matrix((np.asarray(vals, dtype=float), (np.asarray(rows), np.asarray(cols))),
                   shape=(n+m+p+1, nvars)).tocsc()
    start = time.perf_counter()
    try:
        res = milp(c=c, integrality=integrality, bounds=bounds,
                   constraints=LinearConstraint(A, lower, upper),
                   options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap), "presolve": True})
    except Exception as exc:
        return {"status": "error", "status_code": -1, "message": str(exc), "x_values": None,
                "opened_stores": [], "assignments": {}, "solve_seconds": time.perf_counter()-start,
                "optimal": False, "mip_gap": None}
    elapsed = time.perf_counter() - start
    status_names = {0: "Optimal", 1: "Time/iteration limit", 2: "Infeasible", 3: "Unbounded", 4: "Solver error"}
    x = getattr(res, "x", None)
    if x is None or not np.all(np.isfinite(x)):
        return {"status": status_names.get(res.status, "Unknown"), "status_code": int(res.status),
                "message": str(res.message), "x_values": None, "opened_stores": [], "assignments": {},
                "solve_seconds": elapsed, "optimal": False, "mip_gap": getattr(res, "mip_gap", None)}
    x_dec = x[:p] > 0.5
    y_dec = x[p:] > 0.5
    assignment = {}
    for k in np.flatnonzero(x_dec):
        assignment[int(ii[k])] = int(jj[k])
    return {"status": status_names.get(res.status, "Unknown"), "status_code": int(res.status),
            "message": str(res.message), "x_values": x_dec, "y_values": y_dec,
            "opened_stores": list(np.flatnonzero(y_dec)), "assignments": assignment,
            "eligible_customer_indices": ii, "eligible_store_indices": jj,
            "objective_profit": float(-res.fun), "solve_seconds": elapsed,
            "optimal": int(res.status) == 0, "mip_gap": getattr(res, "mip_gap", None),
            "n_eligible_pairs": int(p)}

def build_customer_results(customers, stores, distances, result):
    out = customers.copy()
    out["assigned_store"] = ""
    out["distance_km"] = np.nan
    out["status"] = "Unserved"
    store_ids = stores["store_id"].tolist()
    for i, j in result["assignments"].items():
        out.loc[i, "assigned_store"] = store_ids[j]
        out.loc[i, "distance_km"] = float(distances[i, j])
        out.loc[i, "status"] = "Served"
    return out[["customer_id", "latitude", "longitude", "daily_demand", "assigned_store", "distance_km", "status"]]

def build_store_results(customers, stores, distances, result, revenue_per_order, delivery_cost_per_order_km, capacity_per_store):
    assignments = result["assignments"]
    rows = []
    for j, store in stores.iterrows():
        cust_idx = [i for i, sj in assignments.items() if sj == j]
        orders = int(customers.iloc[cust_idx]["daily_demand"].sum()) if cust_idx else 0
        delivery = float(sum(customers.iloc[i]["daily_demand"] * delivery_cost_per_order_km * distances[i, j] for i in cust_idx))
        gross = float(sum(customers.iloc[i]["daily_demand"] * revenue_per_order for i in cust_idx))
        rows.append({"Store ID": store.store_id, "Latitude": store.latitude, "Longitude": store.longitude,
                     "Assigned customers": len(cust_idx), "Daily orders served": orders,
                     "Capacity": int(capacity_per_store), "Capacity utilization (%)": 100*orders/capacity_per_store,
                     "Revenue (₹)": gross, "Delivery cost (₹)": delivery,
                     "Delivery contribution before fixed cost (₹)": gross-delivery,
                     "Open": bool(j in result["opened_stores"])})
    return pd.DataFrame(rows).query("Open == True").reset_index(drop=True)

def build_kpis(customers, stores, distances, result, max_distance, revenue_per_order,
               delivery_cost_per_order_km, fixed_cost_per_store, capacity_per_store, elapsed=None):
    n = len(customers)
    assignments = result["assignments"]
    served_idx = list(assignments)
    served_orders = int(customers.iloc[served_idx]["daily_demand"].sum()) if served_idx else 0
    total_demand = int(customers["daily_demand"].sum())
    revenue = float(sum(customers.iloc[i]["daily_demand"]*revenue_per_order for i in served_idx))
    delivery = float(sum(customers.iloc[i]["daily_demand"]*delivery_cost_per_order_km*distances[i,j] for i,j in assignments.items()))
    fixed = float(len(result["opened_stores"])*fixed_cost_per_store)
    assigned_distances = [distances[i,j] for i,j in assignments.items()]
    cap_util = served_orders/(len(result["opened_stores"])*capacity_per_store)*100 if result["opened_stores"] else 0.0
    values = [
        ("Solver status", result["status"]), ("Optimality proven", bool(result.get("optimal", False))),
        ("Estimated daily profit", revenue-delivery-fixed), ("Total revenue", revenue),
        ("Total delivery cost", delivery), ("Total fixed operating cost", fixed),
        ("Stores opened", len(result["opened_stores"])), ("Customers served", len(served_idx)),
        ("Customers unserved", n-len(served_idx)), ("Demand served percentage", 100*served_orders/total_demand if total_demand else 0),
        ("Average assigned customer-to-store distance (km)", float(np.mean(assigned_distances)) if assigned_distances else 0.0),
        ("Maximum assigned customer-to-store distance (km)", float(np.max(assigned_distances)) if assigned_distances else 0.0),
        ("Total daily orders served", served_orders), ("Total daily demand", total_demand),
        ("Aggregate open-store capacity utilization (%)", cap_util),
        ("Maximum permitted distance (km)", max_distance),
        ("Eligible customer-store pairs", result.get("n_eligible_pairs", 0)),
        ("Solve time (seconds)", float(elapsed if elapsed is not None else result.get("solve_seconds", 0))),
        ("MIP gap", result.get("mip_gap"))
    ]
    return pd.DataFrame(values, columns=["Metric", "Value"])

def validate_solution(customers, stores, distances, result, max_distance, max_open, capacity_per_store):
    errors = []
    assignments = result.get("assignments", {})
    opened = set(result.get("opened_stores", []))
    if len(opened) > max_open:
        errors.append("maximum number of open stores exceeded")
    loads = {j: 0 for j in range(len(stores))}
    for i, j in assignments.items():
        if j not in opened: errors.append(f"customer index {i} assigned to closed store {j}")
        if distances[i, j] > max_distance + 1e-8: errors.append(f"customer index {i} violates maximum distance")
        loads[j] += int(customers.iloc[i]["daily_demand"])
    for j, load in loads.items():
        if load > capacity_per_store: errors.append(f"store index {j} exceeds capacity")
    return {"valid": not errors, "errors": errors, "loads": loads}

def run_profit_experiment(max_distance=2.0, capacity=1500, revenue=120.0, delivery_cost=8.0,
                          fixed_cost=12000.0, seed=42, time_limit=20, mip_rel_gap=0.02):
    customers, stores = generate_instance(10000, 50, seed, 1, 5)
    distances = haversine_matrix(customers, stores)
    rows = []
    for k in [2, 5, 10, 20, 25]:
        result = solve_location_model(customers, stores, distances, max_distance, k, revenue,
                                      delivery_cost, fixed_cost, capacity, time_limit, mip_rel_gap)
        if result["x_values"] is None:
            rows.append({"Maximum stores allowed (K)": k, "Solver status": result["status"],
                         "Optimality proven": False, "Stores opened": np.nan, "Customers served": np.nan,
                         "Total revenue (₹)": np.nan, "Delivery cost (₹)": np.nan,
                         "Fixed cost (₹)": np.nan, "Estimated daily profit (₹)": np.nan,
                         "Solve time (s)": result["solve_seconds"], "MIP gap": result.get("mip_gap"),
                         "Notes": result.get("message", "")})
            continue
        kpis = build_kpis(customers, stores, distances, result, max_distance, revenue, delivery_cost,
                          fixed_cost, capacity, result["solve_seconds"])
        kv = dict(zip(kpis["Metric"], kpis["Value"]))
        rows.append({"Maximum stores allowed (K)": k, "Solver status": result["status"],
                     "Optimality proven": result["optimal"], "Stores opened": kv["Stores opened"],
                     "Customers served": kv["Customers served"], "Total revenue (₹)": kv["Total revenue"],
                     "Delivery cost (₹)": kv["Total delivery cost"], "Fixed cost (₹)": kv["Total fixed operating cost"],
                     "Estimated daily profit (₹)": kv["Estimated daily profit"],
                     "Solve time (s)": result["solve_seconds"], "MIP gap": result.get("mip_gap"),
                     "Notes": result.get("message", "")})
    return pd.DataFrame(rows)
