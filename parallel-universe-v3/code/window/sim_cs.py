"""Simulation: detection probability of a Cold-Spot-like texture imprint (6 deg) at random positions.
Usage: python3 -I sim_cs.py <sky_code_dir> <planck_dir> <cache_dir> <amp_muK> <nsim> <seed> <out.json>"""
import sys, json, time
sys.path.insert(0, sys.argv[1])
import numpy as np, healpy as hp
import skylib as L
code, pdir, cache, amp, nsim, seed, outp = sys.argv[1:]
amp = float(amp); nsim = int(nsim); seed = int(seed)
st = L.Setup(pdir, cache_dir=cache)
k = st.kinds["texture"]; i6 = k["tc"].index(6)
tau = 24.6
rng = np.random.default_rng(seed); np.random.seed(seed)
rec = []; t0 = time.time()
for s in range(nsim):
    z = rng.uniform(-1, 1); ph = rng.uniform(0, 2 * np.pi)
    n0 = np.array([np.sqrt(1 - z * z) * np.cos(ph), np.sqrt(1 - z * z) * np.sin(ph), z])
    m = st.sim_map(rng) + st.inject_map("texture", i6, [-amp / L.TCMB], n0)
    alm = st.masked_alm(m)
    best = np.zeros(L.NPIX)
    for i, (filt, Ainv) in enumerate(zip(k["filt"], k["Ainv"])):
        b = np.array([hp.alm2map(hp.almxfl(alm, f), L.NSIDE, lmax=L.LMAX) for f in filt])
        d = np.einsum("in,ij,jn->n", b, Ainv, b); d[~st.valid["texture"][i]] = 0
        best = np.maximum(best, d)
    ang = np.degrees(np.arccos(np.clip(st.vec @ n0, -1, 1)))
    near = ang <= 3.0
    pix0 = hp.vec2pix(L.NSIDE, *n0)
    rec.append(dict(det=bool((best[near] > tau).any()), valid=bool(st.valid["texture"][i6][pix0]),
                    dmax_near=float(best[near].max()), false=bool((best[ang > 8] > tau).any())))
json.dump(dict(amp=amp, nsim=nsim, seed=seed, rec=rec, sec=time.time() - t0), open(outp, "w"))
print(amp, nsim, time.time() - t0)
