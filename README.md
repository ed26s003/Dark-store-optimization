# Dark Store Location Optimization — Chennai

A Streamlit MILP app for selecting dark-store sites and assigning customers while enforcing a hard maximum straight-line distance.

## Features
- Sparse binary MILP using `scipy.optimize.milp` (HiGHS backend).
- Reproducible synthetic Chennai-area customer and candidate-site coordinates.
- Hard distance eligibility: pairs beyond `D_max` are excluded from the MILP.
- Customer assignment, store opening, capacity, and maximum-open-store constraints.
- Profit KPIs, interactive map, store/customer CSV exports.
- Required 10,000-customer / 50-candidate experiment for K = 2, 5, 10, 20, 25.
- Unit tests for distance, linkage, capacity, store limits, unserved demand, KPI consistency, reproducibility, and solver statuses.

## Quick start (Python 3.11 recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Mathematical model

Sets: customers `i ∈ I`, candidate stores `j ∈ J`.

Parameters: daily demand `d_i`; capacity `C`; fixed daily store cost `F`; per-order revenue/contribution `r`; delivery cost per order per kilometre `c`; Haversine distance `D_ij`; distance limit `D_max`; store limit `K`.

Variables:
- `x_ij ∈ {0,1}` if customer i is assigned to store j.
- `y_j ∈ {0,1}` if store j is opened.

Maximize:

`sum_(eligible i,j) d_i * (r - c * D_ij) * x_ij - sum_j F * y_j`

Constraints:
1. `sum_j x_ij <= 1`: a customer is served by at most one store.
2. `x_ij <= y_j`: assignments only to open stores.
3. `sum_i d_i*x_ij <= C*y_j`: capacity and opening linkage.
4. `x_ij = 0` for `D_ij > D_max`: implemented by omitting those variables, so the distance rule is enforced in the optimization model itself.
5. `sum_j y_j <= K`: no more than K stores open.
6. `x_ij, y_j` are binary.

The model permits unserved customers. An unserved customer earns no revenue and incurs no delivery cost. Customers with negative per-order contribution will generally remain unserved unless other model assumptions change.

## Distance and synthetic data assumptions
- The app uses Haversine great-circle distance in kilometres.
- Straight-line distance is not road distance and does not encode traffic, one-way streets, or travel time.
- Customer/store coordinates are generated within a Chennai-area bounding box using a fixed random seed. They are simulated points, not real customers, addresses, or businesses.
- Daily demand is synthetic integer order demand. Economic inputs are user-editable and illustrative, not a financial forecast.
- `revenue / contribution per order` is used as the per-order benefit before delivery costs. All reported costs and profit are per day in INR.

## Solver status and experiment interpretation
- `Optimal` means HiGHS reported an optimal solution.
- `Time/iteration limit` with a returned incumbent means a feasible solution was found but global optimality was not proven.
- The experiment records status, solve time, and MIP gap when provided by the solver. A failed solve without an incumbent is reported as missing values, never fabricated.
- For the mathematical optimum, profit is non-decreasing as K increases: the feasible set for K is contained in the feasible set for any larger K. Incumbent profits from separate time-limited runs may not be monotonic because they are not necessarily optimal.
- The required 10,000 x 50 experiment can be compute-intensive. The code uses sparse constraints and only creates assignment variables for distance-eligible pairs. Choose a suitable per-instance time limit for the deployment resources.

## Testing

```bash
python -m unittest discover -s tests -v
```

Tests use small instances and are intended to finish quickly. SciPy/HiGHS version and host resources can affect solve times.

## Deployment at share.streamline.io
The provided hostname is Streamlit Community Cloud's workspace. The standard workflow currently requires a GitHub repository:
1. Create a new GitHub repository and upload the contents of this project (keep `app.py`, `optimizer.py`, `requirements.txt`, `README.md`, and `tests/` at repository root).
2. Sign in at [share.streamline.io](https://share.streamline.io) using the account/GitHub authorization flow.
3. Select **Create app**, choose the repository, branch (typically `main`), and entrypoint `app.py`, then deploy.
4. If offered, select Python 3.11 or another supported version compatible with the pinned dependencies.
5. Wait for build and app startup, review logs if startup fails, then open the generated `.streamlit.app` URL and verify the app.

This project has not been published to your account from this environment; authentication and repository authorization are required for deployment. Do not claim a live URL until the host confirms a successful deployment.

## Project layout
```
app.py
optimizer.py
requirements.txt
README.md
tests/
  test_optimizer.py
```
