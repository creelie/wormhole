"""Drivers for the full-sky search.

python3 -I run.py observe <planck_dir> <data_dir> <out_dir> <map> [cut_deg]
python3 -I run.py null    <planck_dir> <out_dir> <seed> <nsims> [cut_deg]
python3 -I run.py inject  <planck_dir> <out_dir> <seed> <kind> <amp_muK> <ninj> [fixed_size_index]
"""
import sys, json, time
import numpy as np
import healpy as hp
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import skylib as S

mode = sys.argv[1]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def tag(cut):
    return "" if cut == 0 else f"_cut{int(cut)}"


if mode == "observe":
    PD, DD, OUT, name = sys.argv[2:6]
    cut = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    st = S.Setup(PD, extra_cut_deg=cut, cache_dir=OUT); log("setup")
    m = st.data_map(f"{DD}/alm_{name}_l383.fits")
    res = st.scan(st.masked_alm(m), keep_maps=True); log("scan")
    np.savez_compressed(f"{OUT}/obs_{name}{tag(cut)}.npz", map=m.astype(np.float32), mask=st.mask,
                        **{f"{k}_max": v["max"].astype(np.float32) for k, v in res.items()},
                        **{f"{k}_arg": v["arg"] for k, v in res.items()})
    summary = {"map": name, "cut": cut, "fsky": float(st.mask.mean())}
    for k in ("bubble", "sw", "texture"):
        r = res[k]; p = int(np.argmax(r["max"])); kk = "bubble" if k == "sw" else k
        l, b = hp.pix2ang(S.NSIDE, p, lonlat=True); i = int(r["arg"][p])
        entry = {"max": float(r["max"][p]), "lon": float(l), "lat": float(b), "tc": S.BUB_TC[i] if kk == "bubble" else S.TEX_TC[i]}
        if k != "sw":
            entry["amp"] = [float(x) for x in res[kk]["per"][i][1][:, p]]
            entry["err"] = [float(np.sqrt(x)) for x in np.diag(st.kinds[kk]["Ainv"][i])]
        # list of the ten strongest separated peaks
        pk = S.peaks(r["max"], st.good, np.sort(r["max"])[-400], sep_deg=8.0)[:10]
        entry["peaks"] = [{"max": float(r["max"][q]), "lon": float(hp.pix2ang(S.NSIDE, q, lonlat=True)[0]),
                           "lat": float(hp.pix2ang(S.NSIDE, q, lonlat=True)[1]),
                           "tc": (S.BUB_TC if kk == "bubble" else S.TEX_TC)[int(r["arg"][q])]} for q in pk]
        summary[k] = entry
    json.dump(summary, open(f"{OUT}/obs_{name}{tag(cut)}.json", "w"), indent=1)
    log(json.dumps({k: (summary[k]["max"], summary[k]["lon"], summary[k]["lat"], summary[k]["tc"]) for k in ("bubble", "sw", "texture")}))

elif mode == "null":
    PD, OUT, seed, n = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    cut = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    np.random.seed(seed); rng = np.random.default_rng(seed)
    st = S.Setup(PD, extra_cut_deg=cut, cache_dir=OUT); log("setup")
    rec = {k: np.zeros(n) for k in ("bubble", "sw", "texture")}
    loc = {k: np.zeros(n, dtype=int) for k in rec}
    for j in range(n):
        res = st.scan(st.masked_alm(st.sim_map(rng)))
        for k in rec:
            p = int(np.argmax(res[k]["max"])); rec[k][j] = res[k]["max"][p]; loc[k][j] = p
        if (j + 1) % 100 == 0 or j + 1 == 10:
            log(j + 1); np.savez(f"{OUT}/null{tag(cut)}_s{seed}.npz", n=j + 1, **{f"{k}": rec[k][: j + 1] for k in rec}, **{f"{k}_pix": loc[k][: j + 1] for k in rec})
    np.savez(f"{OUT}/null{tag(cut)}_s{seed}.npz", n=n, **rec, **{f"{k}_pix": loc[k] for k in rec})
    log("done")

elif mode == "inject":
    PD, OUT, seed, kind, amp_muK, n = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], float(sys.argv[6]), int(sys.argv[7])
    fixed = int(sys.argv[8]) if len(sys.argv) > 8 else -1
    np.random.seed(seed); rng = np.random.default_rng(seed)
    st = S.Setup(PD, cache_dir=OUT); log("setup")
    tcs = S.BUB_TC if kind == "sw" else S.TEX_TC
    kk = "bubble" if kind == "sw" else "texture"
    # size prior: collisions dN ~ sin(tc) dtc on the grid; textures uniform on the grid
    w = np.sin(np.radians(tcs)) if kind == "sw" else np.ones(len(tcs)); w = w / w.sum()
    rows = []
    for j in range(n):
        i = fixed if fixed >= 0 else int(rng.choice(len(tcs), p=w))
        z = rng.standard_normal(3); n0 = z / np.linalg.norm(z)
        sgn = rng.choice([-1.0, 1.0])
        amps = [sgn * amp_muK / S.TCMB, 0.0] if kind == "sw" else [sgn * amp_muK / S.TCMB]
        m = st.sim_map(rng) + st.inject_map(kk, i, amps, n0)
        res = st.scan(st.masked_alm(m))
        stat = res[kind]["max"]
        ang = np.degrees(np.arccos(np.clip(st.vec @ n0, -1, 1)))
        near = ang <= max(3.0, tcs[i] / 2)
        rows.append([i, float(stat[near].max()) if near.any() else 0.0, float(stat.max()), float(st.frac[tcs[i]][hp.vec2pix(S.NSIDE, *n0)])])
        if (j + 1) % 50 == 0:
            log(j + 1); np.save(f"{OUT}/inj_{kind}_{int(amp_muK)}_f{fixed}_s{seed}.npy", np.array(rows))
    np.save(f"{OUT}/inj_{kind}_{int(amp_muK)}_f{fixed}_s{seed}.npy", np.array(rows))
    log("done")
