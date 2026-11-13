"""Combine nulls, observations and injections into thresholds, p-values, efficiencies and bounds.
Usage: python3 -I analyze.py <out_dir> <result.json>
"""
import sys, json, glob, math
import numpy as np
from scipy import stats

OUT, DEST = sys.argv[1], sys.argv[2]
BUB_TC = [3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 30]
TEX_TC = [1, 2, 3, 4, 5, 6, 7, 8, 10]
MAPS = ["smica", "nilc", "commander", "sevem"]


def load_null(cut=""):
    fs = sorted(glob.glob(f"{OUT}/null{cut}_s*.npz"))
    d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("bubble", "sw", "texture")}
    return d


def poisson_upper(k, cl=0.95):
    """Classical upper limit on a Poisson mean given k observed events (no background)."""
    return 0.5 * stats.chi2.ppf(cl, 2 * (k + 1))


def pval(null, x):
    return (np.sum(null >= x) + 1) / (null.size + 1)


R = {"maps": {}, "stats": {}}
for cut in ("", "_cut20"):
    if not glob.glob(f"{OUT}/null{cut}_s*.npz"):
        continue
    nl = load_null(cut)
    key = "main" if cut == "" else "cut20"
    R["stats"][key] = {}
    for k in ("sw", "bubble", "texture"):
        q = np.quantile(nl[k], [0.5, 0.84, 0.95, 0.99])
        R["stats"][key][k] = {"n": int(nl[k].size), "q50": q[0], "q84": q[1], "tau95": q[2], "tau99": q[3]}
    for mp in MAPS:
        f = f"{OUT}/obs_{mp}{cut}.json"
        try:
            ob = json.load(open(f))
        except FileNotFoundError:
            continue
        R["maps"].setdefault(key, {})[mp] = {}
        for k in ("sw", "bubble", "texture"):
            tau = R["stats"][key][k]["tau95"]
            ncand = sum(1 for p in ob[k]["peaks"] if p["max"] >= tau)
            R["maps"][key][mp][k] = {"max": ob[k]["max"], "lon": ob[k]["lon"], "lat": ob[k]["lat"], "tc": ob[k]["tc"],
                                     "p_global": pval(nl[k], ob[k]["max"]), "ncand95": ncand,
                                     "peaks": ob[k]["peaks"][:6]}

# efficiencies and bounds (main analysis, SMICA)
tau = {k: R["stats"]["main"][k]["tau95"] for k in ("sw", "texture")}
R["eff"] = {}
for kind in ("sw", "texture"):
    rows = {}
    for f in glob.glob(f"{OUT}/inj_{kind}_*_f-1_s*.npy"):
        a = int(f.split("_")[-3]); rows.setdefault(a, []).append(np.load(f))
    ent = []
    k_obs = R["maps"]["main"]["smica"][kind]["ncand95"]
    lam = poisson_upper(k_obs)
    for a in sorted(rows):
        x = np.concatenate(rows[a]); det = x[:, 1] >= tau[kind]; n = len(det); e = det.mean()
        inval = x[:, 3] < 0.85
        e_valid = det[~inval].mean() if (~inval).any() else float("nan")
        ent.append({"amp_muK": a, "n": n, "eff": e, "eff_err": math.sqrt(max(e * (1 - e), 1e-12) / n),
                    "eff_valid": e_valid, "N_upper95": lam / e if e > 0 else None})
    R["eff"][kind] = {"k_obs": k_obs, "lambda95": lam, "rows": ent}

# size-resolved efficiency for collisions
grid = {}
for f in glob.glob(f"{OUT}/inj_sw_*_f[0-9]*_s*.npy"):
    parts = f.split("_"); a = int(parts[-3]); i = int(parts[-2][1:])
    grid.setdefault((i, a), []).append(np.load(f))
R["eff_grid"] = [{"tc": BUB_TC[i], "amp_muK": a, "eff": float((np.concatenate(v)[:, 1] >= tau["sw"]).mean()),
                  "n": int(np.concatenate(v).shape[0])} for (i, a), v in sorted(grid.items())]
json.dump(R, open(DEST, "w"), indent=1, default=float)
print(json.dumps({"stats": R["stats"], "eff_sw": [(r["amp_muK"], round(r["eff"], 3), r["N_upper95"]) for r in R["eff"].get("sw", {}).get("rows", [])],
                  "eff_tex": [(r["amp_muK"], round(r["eff"], 3), r["N_upper95"]) for r in R["eff"].get("texture", {}).get("rows", [])]}, indent=0, default=float))
for key, mm in R["maps"].items():
    for mp, v in mm.items():
        print(key, mp, {k: (round(v[k]["max"], 2), round(v[k]["p_global"], 3), v[k]["ncand95"]) for k in v})
