import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks" / "Pollpred.ipynb"
cell6 = (ROOT / "scripts/pollpred_extras/cell6.py").read_text(encoding="utf-8")

nb = json.loads(nb_path.read_text(encoding="utf-8"))
nb["cells"][33]["source"] = cell6.splitlines(True)
nb["cells"][33]["outputs"] = []
nb["cells"][33]["execution_count"] = None

old = """amp_rows = []
for event_name in events:
    full_model = run_event_did(daily, event_name)
    ns_model = run_event_did(daily_no_spike, event_name)
    did_col = f"{event_name}_did"
    amp_rows.append({
        "event": event_name,
        "coef_full": (np.nan if full_model is None
                      else full_model.params.get(did_col, np.nan)),
        "coef_no_spike": (np.nan if ns_model is None
                          else ns_model.params.get(did_col, np.nan)),
    })"""

new = """_effects_coef = dict(zip(effects["event"], effects["coef"])) if "effects" in globals() else {}

def _demean_twfe(data, col):
    s = data[col].astype(float)
    return (
        s - s.groupby(data["state_id"]).transform("mean")
        - s.groupby(data["date_id"]).transform("mean") + s.mean()
    )

def run_event_did_coef_fast(data, event_name):
    did_col = f"{event_name}_did"
    post_col = f"{event_name}_post"
    if post_col not in data.columns or data[post_col].sum() == 0:
        return np.nan
    y, x = _demean_twfe(data, "log_volume"), _demean_twfe(data, did_col)
    ok = y.notna() & x.notna()
    yv, xv = y[ok].to_numpy(), x[ok].to_numpy()
    denom = float(np.dot(xv, xv))
    return np.nan if denom == 0 else float(np.dot(xv, yv) / denom)

amp_rows = []
for event_name in events:
    amp_rows.append({
        "event": event_name,
        "coef_full": _effects_coef.get(event_name, run_event_did_coef_fast(daily, event_name)),
        "coef_no_spike": run_event_did_coef_fast(daily_no_spike, event_name),
    })"""

src = "".join(nb["cells"][35]["source"])
if old in src:
    nb["cells"][35]["source"] = src.replace(old, new).splitlines(True)
    nb["cells"][35]["outputs"] = []
    print("patched cell 35")
else:
    print("cell 35 pattern not found — skip")

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved cell 33")
