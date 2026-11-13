"""Colour figure set for the v3 paper (Cold Spot part).
Usage: python3 -I figures2c.py <planck_dir> <out_dir> <fig_dir>
"""
import sys, json
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa
from healpy.rotator import Rotator
from matplotlib.colors import LinearSegmentedColormap
_pl = [(0.0, "#0000ff"), (0.25, "#00aaff"), (0.4, "#99e6ff"), (0.5, "#ffffff"), (0.6, "#ffdd44"), (0.75, "#ff7700"), (0.9, "#ee0000"), (1.0, "#880000")]
PLANCK = LinearSegmentedColormap.from_list("planckish", _pl, N=256); PLANCK.set_bad("#bbbbbb")

PD, OUT, FIG = sys.argv[1:4]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 220, "axes.linewidth": 0.8})
NSIDE, LMAX, TCMB = 64, 191, 2.7255e6
CS = (209.0, -57.0)
S = json.load(open(f"{OUT}/result_smica.json")); NI = json.load(open(f"{OUT}/result_nilc.json"))
m = hp.read_map(f"{OUT}/map_smica_n64.fits"); mn = hp.read_map(f"{OUT}/map_nilc_n64.fits")
w = hp.read_map(f"{OUT}/smhw_smica_n64.fits")
mask = (hp.ud_grade(hp.read_map(f"{PD}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"), NSIDE) > 0.99).astype(float)
ell = np.arange(LMAX + 1)
th = np.loadtxt(f"{PD}/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt")
cl = np.zeros(LMAX + 1); L = th[:, 0].astype(int); s = (L >= 2) & (L <= LMAX)
cl[L[s]] = 2 * np.pi * th[s, 1] / (L[s] * (L[s] + 1))
T_l = hp.gauss_beam(np.radians(5 / 60), LMAX) * hp.pixwin(2048)[:LMAX + 1] * hp.gauss_beam(np.radians(2.0), LMAX)
theta_f = np.radians(np.linspace(0, 60, 6001))
def smooth(f):
    bl = hp.beam2bl(f, theta_f, LMAX)
    return np.polynomial.legendre.legval(np.cos(theta_f), bl * (2 * ell + 1) / (4 * np.pi) * T_l)
def bubble_raw(tc, z0, zc):
    tc = np.radians(tc); w1 = (1 - np.cos(theta_f)) / (1 - np.cos(tc))
    return np.where(theta_f < tc, z0 * (1 - w1) + zc * w1, 0.0)
def texture_raw(tc, eps):
    tc = np.radians(tc); ts = np.sqrt(3) / 2 * tc
    g = np.where(theta_f <= ts, 1 / np.sqrt(1 + 4 * (theta_f / tc) ** 2), 0.5 * np.exp(-(theta_f ** 2 - ts ** 2) / (2 * tc ** 2)))
    return eps * g
bb, tt = S["observed"]["bubble"], S["observed"]["texture"]
pb_raw = bubble_raw(bb["tc"], *bb["amp"]) * TCMB; pb = smooth(bubble_raw(bb["tc"], *bb["amp"])) * TCMB
pt_raw = texture_raw(tt["tc"], tt["amp"][0]) * TCMB; pt = smooth(texture_raw(tt["tc"], tt["amp"][0])) * TCMB
vec = np.array(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE)))).T
def model_map(prof, centre):
    c = np.array(hp.pix2vec(NSIDE, centre)); ang = np.arccos(np.clip(vec @ c, -1, 1))
    return np.interp(ang, theta_f, prof)
mb = model_map(pb, bb["centre"]); mt = model_map(pt, tt["centre"])
gray = PLANCK

# 1. full sky
mm = m.copy(); mm[mask == 0] = hp.UNSEEN
hp.mollview(mm, cmap=gray, min=-300, max=300, unit="$\\mu$K", title="Planck 2018 SMICA, $2^\\circ$ FWHM; common mask shown pale", cbar=True, badcolor="#cccccc")
hp.graticule(dpar=30, dmer=60, alpha=0.4)
phi = np.linspace(0, 2 * np.pi, 200)
c0 = hp.ang2vec(CS[0], CS[1], lonlat=True)
e1 = np.cross([0, 0, 1], c0); e1 /= np.linalg.norm(e1); e2 = np.cross(c0, e1)
for rad in (6, 16):
    a_ = np.radians(rad)
    pts = np.cos(a_) * c0[None, :] + np.sin(a_) * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    lo, la = hp.vec2ang(pts, lonlat=True)
    hp.projplot(lo, la, "-" if rad == 16 else "--", color="lime", lonlat=True, lw=1.2 if rad == 16 else 0.7)
plt.savefig(f"{FIG}/fig_fullsky.png", bbox_inches="tight"); plt.close("all")

# 2. Cold Spot region and wavelet (existing style)
plt.figure(figsize=(8, 3.6))
hp.gnomview(m, rot=CS, reso=12, xsize=300, cmap=PLANCK, title="SMICA, $2^\\circ$ FWHM", unit="$\\mu$K", sub=(1, 2, 1), min=-250, max=250, notext=True)
hp.graticule(dpar=10, dmer=10, alpha=0.3)
hp.gnomview(w, rot=CS, reso=12, xsize=300, cmap=PLANCK, title="SMHW coefficient, $R=5^\\circ$", unit="$\\mu$K", sub=(1, 2, 2), notext=True)
hp.graticule(dpar=10, dmer=10, alpha=0.3)
plt.savefig(f"{FIG}/fig_coldspot_map.png", bbox_inches="tight"); plt.close("all")

# 3. template profiles raw and smoothed
fig, axs = plt.subplots(1, 2, figsize=(8, 3.1))
d = np.degrees(theta_f)
axs[0].plot(d, pb_raw, "--", color="#e76f51", lw=1, label="raw profile"); axs[0].plot(d, pb, "-", color="#264653", lw=1.6, label="after $2^\\circ$ smoothing")
axs[0].set_title(f"bubble, $\\theta_c={bb['tc']}^\\circ$"); axs[0].set_xlim(0, 30)
axs[1].plot(d, pt_raw, "--", color="#e76f51", lw=1, label="raw profile"); axs[1].plot(d, pt, "-", color="#2a9d8f", lw=1.6, label="after $2^\\circ$ smoothing")
axs[1].set_title(f"texture, $\\theta_c={tt['tc']}^\\circ$"); axs[1].set_xlim(0, 30)
for a in axs:
    a.axhline(0, color="k", lw=0.4); a.set_xlabel("$\\theta$ [deg]"); a.set_ylabel("$\\Delta T$ [$\\mu$K]"); a.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_templates.png", bbox_inches="tight"); plt.close("all")

# 4. data, models, residuals
plt.figure(figsize=(9, 6.6))
kw = dict(rot=CS, reso=14, xsize=260, cmap=PLANCK, min=-200, max=200, notext=True, cbar=False, margins=(0.01, 0.01, 0.01, 0.07))
for i, (img, ttl) in enumerate([(m, "data"), (mb, "best bubble model"), (m - mb, "data $-$ bubble"),
                                 (m, "data"), (mt, "best texture model"), (m - mt, "data $-$ texture")]):
    hp.gnomview(img, title=ttl, sub=(2, 3, i + 1), **kw); hp.graticule(dpar=10, dmer=10, alpha=0.25)
plt.savefig(f"{FIG}/fig_bestfit.png", bbox_inches="tight"); plt.close("all")

# 5. delta chi2 vs size
fig, axs = plt.subplots(1, 2, figsize=(8, 3.1))
for a, kind in zip(axs, ("bubble", "texture")):
    for R_, st, lab in ((S, "-o", "SMICA"), (NI, "--s", "NILC")):
        rows = [r for r in R_["table"] if r["kind"] == kind]
        a.plot([r["tc"] for r in rows], [r["dchi2"] for r in rows], st, ms=4, color=("#d62728" if lab=="SMICA" else "#1f77b4"), mfc="white", label=lab)
    q = np.array(S["sim_quantiles"][f"{kind}_sel"])
    a.axhline(q[2], color="#2ca02c", lw=0.9, ls=":", label="95% point, selected null")
    a.set_xlabel("$\\theta_c$ [deg]"); a.set_ylabel("$\\Delta\\chi^2$ at best centre"); a.set_title(kind); a.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_dchi_size.png", bbox_inches="tight"); plt.close("all")

# 6. ring profile with smoothed models
def rings(mp, centre):
    c = np.array(hp.pix2vec(NSIDE, centre)); ang = np.degrees(np.arccos(np.clip(vec @ c, -1, 1)))
    b = np.arange(0, 31, 1.0); return 0.5 * (b[:-1] + b[1:]), np.array([mp[(ang >= x) & (ang < y)].mean() for x, y in zip(b[:-1], b[1:])])
fig, ax = plt.subplots(figsize=(5.6, 3.4))
x, y = rings(m, tt["centre"]); ax.plot(x, y, "o", color="#2a9d8f", ms=3.5, mfc="white", label="SMICA ring mean (texture centre)")
x2, y2 = rings(mb, bb["centre"]); x3, y3 = rings(mt, tt["centre"])
x4, y4 = rings(m, bb["centre"]); ax.plot(x4, y4, "^", color="#e76f51", ms=3.5, mfc="#f4a261", label="SMICA ring mean (bubble centre)")
ax.plot(x2, y2, "-", color="#d62728", lw=1.6, label="bubble model"); ax.plot(x3, y3, "--", color="#1f77b4", lw=1.4, label="texture model")
ax.axhline(0, color="k", lw=0.4); ax.set_xlabel("angle from centre [deg]"); ax.set_ylabel("$\\Delta T$ [$\\mu$K]"); ax.legend(frameon=False, fontsize=7.5)
plt.savefig(f"{FIG}/fig_profile.png", bbox_inches="tight"); plt.close("all")

# 7. 3D surface of the Cold Spot
img = hp.gnomview(m, rot=CS, reso=15, xsize=160, return_projected_map=True, no_plot=True)
plt.close("all")
fig = plt.figure(figsize=(6.4, 4.4)); ax = fig.add_subplot(111, projection="3d")
n = img.shape[0]; X, Y = np.meshgrid((np.arange(n) - n / 2) * 15 / 60, (np.arange(n) - n / 2) * 15 / 60)
ax.plot_surface(X, Y, np.asarray(img), cmap="coolwarm", rstride=2, cstride=2, lw=0.05, edgecolor="#333333", antialiased=True)
ax.set_xlabel("x [deg]"); ax.set_ylabel("y [deg]"); ax.set_zlabel("$\\Delta T$ [$\\mu$K]"); ax.view_init(elev=32, azim=-60)
ax.set_title("Cold Spot as a temperature surface (SMICA, $40^\\circ\\times40^\\circ$)")
plt.savefig(f"{FIG}/fig_surface3d.png", bbox_inches="tight"); plt.close("all")

# 8. 3D schematic of the collision geometry
fig = plt.figure(figsize=(6, 4.6)); ax = fig.add_subplot(111, projection="3d")
u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#8ecae6", lw=0.3)
tc = np.radians(35); uu, vv = np.mgrid[0:2 * np.pi:60j, 0:tc:15j]
ax.plot_surface(np.cos(vv), np.sin(vv) * np.cos(uu), np.sin(vv) * np.sin(uu), color="#e63946", alpha=0.85, lw=0)
ph = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(tc) * np.ones_like(ph), np.sin(tc) * np.cos(ph), np.sin(tc) * np.sin(ph), color="#9d0208", lw=2)
R2, d0 = 1.9, 2.45; u2, v2 = np.mgrid[0:2 * np.pi:40j, np.pi - 0.62:np.pi:10j]
ax.plot_surface(d0 + R2 * np.cos(v2), R2 * np.sin(v2) * np.cos(u2), R2 * np.sin(v2) * np.sin(u2), color="#ffb703", alpha=0.5, lw=0)
ax.scatter([0], [0], [0], color="k", s=20)
ax.plot([0, np.cos(tc)], [0, np.sin(tc)], [0, 0], "k:", lw=1); ax.plot([0, 1], [0, 0], [0, 0], "k:", lw=1)
ax.set_box_aspect((1.3, 1, 1), zoom=1.45); ax.set_axis_off(); ax.view_init(elev=14, azim=-58)
ax.set_xlim(-1, 1.6); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
fig.text(0.47, 0.44, "O", fontsize=9)
fig.text(0.24, 0.84, "last-scattering sphere", fontsize=8)
fig.text(0.70, 0.66, "collision disc", fontsize=8)
fig.text(0.68, 0.17, "wall of the\nneighbouring bubble", fontsize=8)
plt.savefig(f"{FIG}/fig_schematic.png", bbox_inches="tight"); plt.close("all")

# 9. conformal diagram of two colliding bubbles
from matplotlib.patches import Rectangle, Polygon
fig, ax = plt.subplots(figsize=(5.4, 4.4))
box = Rectangle((-1, -0.82), 2, 1.82, fill=False, ec="k", lw=0.8); ax.add_patch(box)
ax.text(0, 1.05, "future infinity of the false vacuum", ha="center", fontsize=8)
xL, xR, t0 = -0.45, 0.55, -0.55; xc_, yc_ = 0.05, -0.05
ax.add_patch(Polygon([(xc_, yc_), (xc_, 1), (xc_ - (1 - yc_), 1)], closed=True, fc="0.82", ec="none", zorder=0))
lines = [((xL, t0), (xL - 1.6, t0 + 1.6)), ((xL, t0), (xc_, yc_)), ((xR, t0), (xc_, yc_)), ((xR, t0), (xR + 1.6, t0 + 1.6))]
for (p, q_) in lines:
    l, = ax.plot([p[0], q_[0]], [p[1], q_[1]], "-", color="#e76f51", lw=1.6); l.set_clip_path(box)
ax.plot([xc_, xc_], [yc_, 1], "--", color="#6a4c93", lw=1.2)
l, = ax.plot([xc_, xc_ - (1 - yc_)], [yc_, 1], color="0.35", lw=0.8, ls=":"); l.set_clip_path(box)
for x_ in (xL, xR):
    ax.plot([x_], [t0], "o", color="#e76f51", ms=5); ax.text(x_, t0 - 0.13, "nucleation", ha="center", fontsize=7)
ax.plot([xc_], [yc_], "s", color="#6a4c93", ms=5); ax.text(xc_ + 0.05, yc_ - 0.1, "collision", fontsize=7)
ax.text(-0.45, -0.32, "our bubble", ha="center", fontsize=8.5)
ax.text(0.58, 0.2, "neighbouring\nbubble", ha="center", fontsize=8.5)
ax.text(xc_ + 0.04, 0.82, "domain\nwall", fontsize=7)
ox, oy, ls_ = -0.5, 0.8, 0.45
ax.plot([ox, ox - (oy - ls_)], [oy, ls_], "-", color="#2a9d8f", lw=0.9); ax.plot([ox, ox + (oy - ls_)], [oy, ls_], "-", color="#2a9d8f", lw=0.9)
ax.plot([ox - (oy - ls_), ox + (oy - ls_)], [ls_, ls_], "k:", lw=1.3)
ax.plot([-ls_, ox + (oy - ls_)], [ls_, ls_], "-", color="#f4a261", lw=3.4)
ax.plot([ox], [oy], "k*", ms=9)
ax.text(ox - 0.05, oy + 0.07, "observer", ha="center", fontsize=7)
ax.annotate("part of the last-scattering\nsurface inside the collision's\nfuture: a disc on our sky", xy=(-0.3, ls_), xytext=(-0.98, 0.02), fontsize=6.5,
            arrowprops=dict(arrowstyle="->", lw=0.6))
ax.annotate("causal future\nof the collision", xy=(-0.08, 0.62), xytext=(0.3, 0.62), fontsize=6.5, va="center", arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.9, 1.2); ax.set_aspect("equal"); ax.set_axis_off()
plt.savefig(f"{FIG}/fig_conformal.png", bbox_inches="tight"); plt.close("all")

# 10. analysis pipeline
fig, ax = plt.subplots(figsize=(8.6, 3.6)); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)
boxes = [(0.1, 2.9, "Planck PR3\nSMICA / NILC\n$N_{\\rm side}=2048$"), (2.1, 2.9, "harmonics to\n$\\ell=191$, $2^\\circ$ beam,\n$N_{\\rm side}=64$"),
         (4.1, 2.9, "$30^\\circ$ region,\nmask, covariance\n$\\mathbf{C}$ (Legendre sum)"), (6.1, 2.9, "templates:\n1620 bubble,\n1215 texture\n(size $\\times$ centre)"),
         (8.1, 2.9, "max $\\Delta\\chi^2$,\nBayes factor"),
         (0.1, 0.4, "Planck best-fit\n$C_\\ell$"), (2.1, 0.4, "Gaussian skies\n(full sky + patch)"), (4.1, 0.4, "coldest SMHW spot,\nrotate to Cold Spot"),
         (6.1, 0.4, "same fits\non every sky"), (8.1, 0.4, "null distribution,\n$p=(r+1)/(N+1)$")]
for x, y, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 1.75, 1.05, boxstyle="round,pad=0.05", fc="#e0fbfc", ec="#3d5a80", lw=0.9))
    ax.text(x + 0.875, y + 0.52, t, ha="center", va="center", fontsize=7.2)
for y in (3.42, 0.92):
    for x in (1.9, 3.9, 5.9, 7.9):
        ax.annotate("", xy=(x + 0.17, y), xytext=(x - 0.03, y), arrowprops=dict(arrowstyle="->", lw=0.9))
ax.annotate("", xy=(9.0, 2.85), xytext=(9.0, 1.5), arrowprops=dict(arrowstyle="->", lw=0.9))
ax.text(9.1, 2.15, "compare", fontsize=7)
ax.annotate("", xy=(4.95, 2.85), xytext=(1.0, 1.5), arrowprops=dict(arrowstyle="->", lw=0.6, ls="--"))
plt.savefig(f"{FIG}/fig_pipeline.png", bbox_inches="tight"); plt.close("all")

# 11. selection illustrated on one Gaussian sky
rng_seed = 4
np.random.seed(rng_seed)
alm = hp.synalm(cl * T_l ** 2, lmax=LMAX, new=True)
sim = hp.alm2map(alm, NSIDE, lmax=LMAX)
def smhw_profile(R):
    y = 2 * np.tan(theta_f / 2); N = np.sqrt(1 + R ** 2 / 2 + R ** 4 / 4)
    return (1 / (np.sqrt(2 * np.pi) * R * N)) * (1 + (y / 2) ** 2) ** 2 * (2 - (y / R) ** 2) * np.exp(-y ** 2 / (2 * R ** 2))
wl = hp.beam2bl(smhw_profile(np.radians(5)), theta_f, LMAX)
ws = hp.alm2map(hp.almxfl(hp.map2alm(sim * mask, lmax=LMAX, iter=0), wl), NSIDE, lmax=LMAX)
wm = hp.alm2map(hp.almxfl(hp.map2alm(mask, lmax=LMAX), hp.gauss_beam(np.radians(10), LMAX)), NSIDE, lmax=LMAX)
good = (wm > 0.95) & (mask > 0)
pmin = np.flatnonzero(good)[np.argmin(ws[good])]
lon, lat = hp.pix2ang(NSIDE, pmin, lonlat=True)
a2 = Rotator(rot=[CS[0], CS[1], 0], deg=True, inv=True).rotate_alm(Rotator(rot=[lon, lat, 0], deg=True).rotate_alm(hp.map2alm(sim, lmax=LMAX, iter=3)))
simr = hp.alm2map(a2, NSIDE, lmax=LMAX)
plt.figure(figsize=(9, 2.9))
s1 = sim.copy(); s1[mask == 0] = hp.UNSEEN
hp.mollview(s1, cmap=gray, min=-300, max=300, sub=(1, 2, 1), title="a Gaussian $\\Lambda$CDM sky; coldest wavelet spot marked", cbar=False)
hp.projscatter(lon, lat, lonlat=True, marker="o", s=90, facecolors="none", edgecolors="lime", linewidths=1.8)
s2 = simr.copy(); s2[mask == 0] = hp.UNSEEN
hp.mollview(s2, cmap=gray, min=-300, max=300, sub=(1, 2, 2), title="same sky rotated: spot moved to $(209^\\circ,-57^\\circ)$", cbar=False)
hp.projscatter(CS[0], CS[1], lonlat=True, marker="o", s=90, facecolors="none", edgecolors="lime", linewidths=1.8)
plt.savefig(f"{FIG}/fig_selection.png", bbox_inches="tight"); plt.close("all")

# 12. null distributions of max dchi2 and of lnB
sims = np.load(f"{OUT}/sims_smica.npz")
fig, axs = plt.subplots(2, 2, figsize=(8, 5.6))
for j, kind in enumerate(("bubble", "texture")):
    a = axs[0, j]; sel = sims[f"{kind}_sel"]; fix = sims[f"{kind}_fixed"]; ob = S["observed"][kind]["dchi2"]
    b = np.linspace(0, max(sel.max(), ob) * 1.05, 45)
    a.hist(fix, bins=b, histtype="stepfilled", color="#a8dadc", ec="#457b9d", label="fixed position")
    a.hist(sel, bins=b, histtype="step", color="#e76f51", lw=1.4, label="coldest spot selected")
    a.axvline(ob, color="#d62728", lw=2); a.set_title(f"{kind}: max $\\Delta\\chi^2$"); a.legend(frameon=False, fontsize=7)
    a = axs[1, j]; k = 2 if kind == "bubble" else 1
    lb = sims[f"lnB_{kind}_sel"] - k * np.log(2); obB = S["observed"][kind]["lnB"] - k * np.log(2)
    b = np.linspace(min(lb.min(), obB) - 0.2, max(lb.max(), obB) + 0.2, 45)
    a.hist(lb, bins=b, histtype="stepfilled", color="#bde0fe", ec="#1d3557", lw=1.0); a.axvline(obB, color="#d62728", lw=2)
    a.set_title(f"{kind}: $\\ln B$, coldest spot selected")
for a in axs.flat: a.set_ylabel("simulations")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_null.png", bbox_inches="tight"); plt.close("all")

# 13. SMICA vs NILC in the region
cpix = hp.query_disc(NSIDE, hp.ang2vec(*CS, lonlat=True), np.radians(30)); cpix = cpix[mask[cpix] > 0]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
axs[0].plot(m[cpix], mn[cpix], "k.", ms=1.5); lim = 300
axs[0].plot([-lim, lim], [-lim, lim], "k-", lw=0.5); axs[0].set_xlim(-lim, lim); axs[0].set_ylim(-lim, lim)
axs[0].set_xlabel("SMICA [$\\mu$K]"); axs[0].set_ylabel("NILC [$\\mu$K]"); axs[0].set_title(f"{len(cpix)} pixels of the region")
axs[1].hist(m[cpix] - mn[cpix], bins=50, histtype="stepfilled", color="#90be6d", ec="#386641")
axs[1].set_xlabel("SMICA $-$ NILC [$\\mu$K]"); axs[1].set_title(f"rms difference {np.std(m[cpix]-mn[cpix]):.1f} $\\mu$K")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_smica_nilc.png", bbox_inches="tight"); plt.close("all")

# 14. spherical Mexican-hat wavelet, 3D and harmonic response
fig = plt.figure(figsize=(8.4, 3.4))
ax = fig.add_subplot(121, projection="3d")
g = np.linspace(-15, 15, 121); X, Y = np.meshgrid(g, g); TH = np.radians(np.hypot(X, Y))
Rr = np.radians(5); yv = 2 * np.tan(TH / 2); Nn = np.sqrt(1 + Rr ** 2 / 2 + Rr ** 4 / 4)
Z = (1 / (np.sqrt(2 * np.pi) * Rr * Nn)) * (1 + (yv / 2) ** 2) ** 2 * (2 - (yv / Rr) ** 2) * np.exp(-yv ** 2 / (2 * Rr ** 2))
ax.plot_surface(X, Y, Z, cmap="viridis", rstride=3, cstride=3, lw=0.1, edgecolor="k"); ax.view_init(28, -55)
ax.set_title("SMHW kernel, $R=5^\\circ$"); ax.set_xlabel("deg"); ax.set_ylabel("deg"); ax.set_zticks([])
ax2 = fig.add_subplot(122)
ax2.plot(ell[2:], wl[2:] / np.abs(wl).max(), "-", color="#7209b7"); ax2.set_xlabel("$\\ell$"); ax2.set_ylabel("$\\psi_\\ell$ (normalised)")
ax2.set_title("harmonic response: a band-pass filter"); ax2.axhline(0, color="k", lw=0.4)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_wavelet.png", bbox_inches="tight"); plt.close("all")

# 15. power spectrum and transfer
fig, ax = plt.subplots(figsize=(5.6, 3.3))
Dl = ell * (ell + 1) * cl / (2 * np.pi)
ax.plot(ell[2:], Dl[2:], "-", color="#d62728", lw=1.5, label="Planck 2018 best fit $D_\\ell$")
ax.plot(ell[2:], (Dl * T_l ** 2)[2:], "--", color="#1f77b4", lw=1.3, label="after transfer $T_\\ell^2$")
ax.set_xlabel("$\\ell$"); ax.set_ylabel("$D_\\ell=\\ell(\\ell+1)C_\\ell/2\\pi$ [$\\mu$K$^2$]"); ax.legend(frameon=False, fontsize=8)
plt.savefig(f"{FIG}/fig_spectrum.png", bbox_inches="tight"); plt.close("all")
from PIL import Image, ImageChops
import glob as _g
for f in _g.glob(f"{FIG}/*.png"):
    im = Image.open(f); bg = Image.new(im.mode, im.size, (255, 255, 255, 255) if im.mode == "RGBA" else 255 if im.mode == "L" else (255, 255, 255))
    bb_ = ImageChops.difference(im.convert("RGB"), bg.convert("RGB")).getbbox()
    if bb_: im.crop((max(bb_[0] - 12, 0), max(bb_[1] - 12, 0), min(bb_[2] + 12, im.width), min(bb_[3] + 12, im.height))).save(f)
print("all figures written")
