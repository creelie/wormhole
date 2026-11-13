"""Planck PR3 test of bubble-collision and texture templates at the CMB Cold Spot.

Usage: python3 -I coldspot.py <planck_dir> <out_dir> <map_name> <nsims> <seed>
"""
import sys, json, time
import numpy as np
import healpy as hp
from scipy.linalg import cho_factor, cho_solve
from scipy.special import eval_legendre
from scipy.stats import kurtosis

PD, OUT, MAPNAME, NSIMS, SEED = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
NSIDE = 64
LMAX = 3 * NSIDE - 1
FWHM = np.radians(2.0)
TCMB = 2.7255e6  # muK
CS_L, CS_B = 209.0, -57.0           # Cold Spot centre (Cruz et al. 2005)
PATCH_R = np.radians(30.0)
SCAN_R = np.radians(6.0)
NOISE = 1.0                          # muK white noise per pixel in the model
BUB_TC = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]
TEX_TC = [1, 2, 3, 4, 5, 6, 7, 8, 10]
SMHW_R = np.radians(5.0)
rng = np.random.default_rng(SEED)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ---------- theory spectrum and transfer function ----------
th = np.loadtxt(f"{PD}/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt")
cl = np.zeros(LMAX + 1)
L = th[:, 0].astype(int)
sel = (L >= 2) & (L <= LMAX)
cl[L[sel]] = 2 * np.pi * th[sel, 1] / (L[sel] * (L[sel] + 1))   # muK^2
bl_native = hp.gauss_beam(np.radians(5 / 60), LMAX) * hp.pixwin(2048)[: LMAX + 1]
gl = hp.gauss_beam(FWHM, LMAX)
T_l = bl_native * gl
clT = cl * T_l ** 2

# ---------- data ----------
def load_map(name):
    m = hp.read_map(f"{PD}/COM_CMB_IQU-{name}_2048_R3.00_full.fits", field=0) * 1e6  # muK
    alm = hp.map2alm(m, lmax=LMAX, use_pixel_weights=True)
    fl = gl.copy(); fl[:2] = 0
    alm = hp.almxfl(alm, fl)
    return hp.alm2map(alm, NSIDE, lmax=LMAX)

dmap = load_map(MAPNAME)
mask_hi = hp.read_map(f"{PD}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")
mask = hp.ud_grade(mask_hi, NSIDE)
mask = (mask > 0.99).astype(float)
log("data loaded; fsky", mask.mean())

# ---------- patch geometry ----------
cs_vec = hp.ang2vec(CS_L, CS_B, lonlat=True)
pix_patch = hp.query_disc(NSIDE, cs_vec, PATCH_R)
pix_patch = pix_patch[mask[pix_patch] > 0]
vecs = np.array(hp.pix2vec(NSIDE, pix_patch)).T
npx = len(pix_patch)
pix_scan = hp.query_disc(NSIDE, cs_vec, SCAN_R)
cen_vecs = np.array(hp.pix2vec(NSIDE, pix_scan)).T
log("patch pixels", npx, "scan centres", len(pix_scan))

# covariance from Legendre sum
cosang = np.clip(vecs @ vecs.T, -1, 1)
ell = np.arange(LMAX + 1)
coef = (2 * ell + 1) / (4 * np.pi) * clT
Cmat = np.polynomial.legendre.legval(cosang, coef) + NOISE ** 2 * np.eye(npx)
cf = cho_factor(Cmat)
Lchol = np.linalg.cholesky(Cmat)
log("covariance ready")

# ---------- smoothed template profiles ----------
theta_f = np.radians(np.linspace(0, 60, 6001))
def smooth_profile(f):
    """Convolve an azimuthally symmetric profile f(theta) with the transfer T_l."""
    bl = hp.beam2bl(f, theta_f, LMAX)            # f_l * 2pi/(2l+1) style coefficients
    flm = bl * (2 * ell + 1) / (4 * np.pi) * T_l
    return np.polynomial.legendre.legval(np.cos(theta_f), flm)

def bubble_profiles(tc):
    tc = np.radians(tc)
    inside = theta_f < tc
    f0 = np.where(inside, 1 - (1 - np.cos(theta_f)) / (1 - np.cos(tc)), 0.0)  # weight of z0
    f1 = np.where(inside, (1 - np.cos(theta_f)) / (1 - np.cos(tc)), 0.0)      # weight of z_crit
    return smooth_profile(f0) * TCMB, smooth_profile(f1) * TCMB

def texture_profile(tc):
    tc = np.radians(tc)
    ts = np.sqrt(3) / 2 * tc
    g = 1 / np.sqrt(1 + 4 * (theta_f / tc) ** 2)
    gs = 1 / np.sqrt(1 + 4 * (ts / tc) ** 2)
    g = np.where(theta_f <= ts, g, gs * np.exp(-(theta_f ** 2 - ts ** 2) / (2 * tc ** 2)))
    return smooth_profile(g) * TCMB

def templates_at(cen, profs):
    ang = np.arccos(np.clip(vecs @ cen, -1, 1))
    return np.stack([np.interp(ang, theta_f, p) for p in profs])

models = []   # (kind, tc, centre index, T (k x npx), A^{-1}, CinvT)
for kind, grid in (("bubble", BUB_TC), ("texture", TEX_TC)):
    for tc in grid:
        profs = bubble_profiles(tc) if kind == "bubble" else (texture_profile(tc),)
        for ci, cen in enumerate(cen_vecs):
            Tm = templates_at(cen, profs)
            CiT = cho_solve(cf, Tm.T).T
            A = Tm @ CiT.T
            models.append((kind, tc, ci, Tm, np.linalg.inv(A), CiT, np.linalg.slogdet(A)[1]))
log("templates", len(models))

# stacked arrays for fast evaluation
KIND = {}
for kind in ("bubble", "texture"):
    idx = [i for i, m in enumerate(models) if m[0] == kind]
    KIND[kind] = dict(idx=np.array(idx),
                      CiT=np.stack([models[i][5] for i in idx]),     # (N,k,npx)
                      Ai=np.stack([models[i][4] for i in idx]),      # (N,k,k)
                      logdet=np.array([models[i][6] for i in idx]))
    KIND[kind]["CiT2"] = KIND[kind]["CiT"].reshape(-1, npx)

def fit_kind(kind, d):
    K = KIND[kind]
    N, k, _ = K["CiT"].shape
    b = (K["CiT2"] @ d).reshape(N, k)
    a = np.einsum("nkl,nl->nk", K["Ai"], b)
    return np.sum(a * b, axis=1), a

def fit_all(d):
    out = [None] * len(models)
    for kind in KIND:
        dc, a = fit_kind(kind, d)
        for j, i in enumerate(KIND[kind]["idx"]):
            out[i] = (float(dc[j]), a[j])
    return out

def summarize(d, want_res=False):
    s = {}
    for kind in KIND:
        K = KIND[kind]
        dc, a = fit_kind(kind, d)
        j = int(np.argmax(dc)); i = K["idx"][j]
        k = a.shape[1]
        lnB = 0.5 * dc + 0.5 * k * np.log(2 * np.pi) - 0.5 * K["logdet"] - k * np.log(2e-4)
        s[kind] = dict(dchi2=float(dc[j]), tc=models[i][1], centre=int(pix_scan[models[i][2]]),
                       amp=[float(v) for v in a[j]],
                       lnB=float(np.log(np.mean(np.exp(lnB - lnB.max()))) + lnB.max()))
    return s, (fit_all(d) if want_res else None)

# ---------- observed ----------
dpatch = dmap[pix_patch]
obs, obs_res = summarize(dpatch, want_res=True)
log("observed", json.dumps(obs))

# per-model observed table for the paper
tab = []
for (kind, tc, ci, Tm, Ai, CiT, A), (dc, a) in zip(models, obs_res):
    if pix_scan[ci] == obs[kind]["centre"]:
        err = np.sqrt(np.diag(Ai))
        tab.append(dict(kind=kind, tc=tc, dchi2=dc, amp=[float(v) for v in a], err=[float(v) for v in err]))

# ---------- SMHW on the full sky ----------
def smhw_profile(R):
    y = 2 * np.tan(theta_f / 2)
    N = np.sqrt(1 + R ** 2 / 2 + R ** 4 / 4)
    return (1 / (np.sqrt(2 * np.pi) * R * N)) * (1 + (y / 2) ** 2) ** 2 * (2 - (y / R) ** 2) * np.exp(-y ** 2 / (2 * R ** 2))
wl = hp.beam2bl(smhw_profile(SMHW_R), theta_f, LMAX)
wmask = hp.alm2map(hp.almxfl(hp.map2alm(mask, lmax=LMAX), hp.gauss_beam(np.radians(10), LMAX)), NSIDE, lmax=LMAX)
wmask = (wmask > 0.95) & (mask > 0)
def smhw_stats(m):
    alm = hp.map2alm(m * mask, lmax=LMAX, iter=0)
    w = hp.alm2map(hp.almxfl(alm, wl), NSIDE, lmax=LMAX)
    v = w[wmask]
    return w, float(v.min()), float(kurtosis(v)), int(np.flatnonzero(wmask)[np.argmin(v)])
wobs, wmin_obs, kurt_obs, wpix_obs = smhw_stats(dmap)
cs_pix = hp.vec2pix(NSIDE, *cs_vec)
log("SMHW obs min", wmin_obs, "kurt", kurt_obs, "at", hp.pix2ang(NSIDE, wpix_obs, lonlat=True))

# ---------- simulations ----------
from healpy.rotator import Rotator
sims = dict(bubble_fixed=[], texture_fixed=[], bubble_sel=[], texture_sel=[], lnB_bubble=[], lnB_texture=[], lnB_bubble_sel=[], lnB_texture_sel=[],
            wmin=[], kurt=[])
for s in range(NSIMS):
    # (a) fixed location, exact patch covariance
    d = Lchol @ rng.standard_normal(npx)
    ss, _ = summarize(d)
    sims["bubble_fixed"].append(ss["bubble"]["dchi2"]); sims["texture_fixed"].append(ss["texture"]["dchi2"])
    sims["lnB_bubble"].append(ss["bubble"]["lnB"]); sims["lnB_texture"].append(ss["texture"]["lnB"])
    # (b) full-sky sim, select coldest SMHW spot, rotate it to the Cold Spot position, then fit
    alm = hp.synalm(clT, lmax=LMAX, new=True)
    m = hp.alm2map(alm, NSIDE, lmax=LMAX) + NOISE * rng.standard_normal(hp.nside2npix(NSIDE))
    w, wmin, ku, wpix = smhw_stats(m)
    sims["wmin"].append(wmin); sims["kurt"].append(ku)
    lon, lat = hp.pix2ang(NSIDE, wpix, lonlat=True)
    alm_n = hp.map2alm(m, lmax=LMAX, iter=3)
    r1 = Rotator(rot=[lon, lat, 0], deg=True, inv=False)      # spot -> (0,0)
    r2 = Rotator(rot=[CS_L, CS_B, 0], deg=True, inv=True)     # (0,0) -> Cold Spot
    alm_r = r2.rotate_alm(r1.rotate_alm(alm_n))
    mr = hp.alm2map(alm_r, NSIDE, lmax=LMAX)
    ss2, _ = summarize(mr[pix_patch])
    sims["bubble_sel"].append(ss2["bubble"]["dchi2"]); sims["texture_sel"].append(ss2["texture"]["dchi2"])
    sims["lnB_bubble_sel"].append(ss2["bubble"]["lnB"]); sims["lnB_texture_sel"].append(ss2["texture"]["lnB"])
    if s % 50 == 0:
        log("sim", s)
        np.savez(f"{OUT}/sims_{MAPNAME}_partial.npz", **{k: np.array(v) for k, v in sims.items()})

pval = lambda arr, v, greater=True: float((np.sum(np.array(arr) >= v) + 1) / (len(arr) + 1)) if greater else float((np.sum(np.array(arr) <= v) + 1) / (len(arr) + 1))
result = dict(
    map=MAPNAME, nside=NSIDE, fwhm_deg=2.0, patch_pixels=npx, nsims=NSIMS, seed=SEED,
    observed=obs, table=tab,
    smhw=dict(R_deg=5.0, min_obs=wmin_obs, kurt_obs=kurt_obs, min_pix_lonlat=list(map(float, hp.pix2ang(NSIDE, wpix_obs, lonlat=True))),
              p_min=pval(sims["wmin"], wmin_obs, greater=False), p_kurt=pval(sims["kurt"], kurt_obs)),
    p_fixed=dict(bubble=pval(sims["bubble_fixed"], obs["bubble"]["dchi2"]), texture=pval(sims["texture_fixed"], obs["texture"]["dchi2"])),
    p_selected=dict(bubble=pval(sims["bubble_sel"], obs["bubble"]["dchi2"]), texture=pval(sims["texture_sel"], obs["texture"]["dchi2"])),
    p_lnB=dict(bubble=pval(sims["lnB_bubble"], obs["bubble"]["lnB"]), texture=pval(sims["lnB_texture"], obs["texture"]["lnB"])),
    p_lnB_selected=dict(bubble=pval(sims["lnB_bubble_sel"], obs["bubble"]["lnB"]), texture=pval(sims["lnB_texture_sel"], obs["texture"]["lnB"])),
    sim_quantiles={k: [float(q) for q in np.percentile(v, [50, 84, 95, 99])] for k, v in sims.items()},
)
json.dump(result, open(f"{OUT}/result_{MAPNAME}.json", "w"), indent=1)
np.savez(f"{OUT}/sims_{MAPNAME}.npz", **{k: np.array(v) for k, v in sims.items()})
hp.write_map(f"{OUT}/map_{MAPNAME}_n64.fits", dmap, overwrite=True, dtype=np.float32)
hp.write_map(f"{OUT}/smhw_{MAPNAME}_n64.fits", wobs, overwrite=True, dtype=np.float32)
log("done", json.dumps({k: result[k] for k in ("p_fixed", "p_selected", "p_lnB", "smhw")}))
