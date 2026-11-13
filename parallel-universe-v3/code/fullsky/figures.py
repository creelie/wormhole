"""Colour figures for the full-sky search paper.
Usage: python3 -I figures.py <planck_dir> <out_dir> <result.json or -> <fig_dir> [which ...]
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import skylib as S

PD, OUT, RES, FIG = sys.argv[1:5]
WHICH = set(sys.argv[5:])
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "axes.linewidth": 0.8, "savefig.dpi": 220})
MAPS = ["smica", "nilc", "commander", "sevem"]
MCOL = {"smica": "#d62728", "nilc": "#1f77b4", "commander": "#2ca02c", "sevem": "#9467bd"}
MLAB = {"smica": "SMICA", "nilc": "NILC", "commander": "Commander", "sevem": "SEVEM"}

# Planck-like diverging colour map
_pl = [(0.0, "#0000ff"), (0.25, "#00aaff"), (0.4, "#99e6ff"), (0.5, "#ffffff"), (0.6, "#ffdd44"),
       (0.75, "#ff7700"), (0.9, "#ee0000"), (1.0, "#880000")]
PLANCK = LinearSegmentedColormap.from_list("planckish", _pl, N=256)
PLANCK.set_bad("#bbbbbb"); PLANCK.set_under("white")


def want(name):
    return not WHICH or name in WHICH


def save(name):
    p = f"{FIG}/{name}.png"
    plt.savefig(p, bbox_inches="tight", pad_inches=0.03)
    plt.close("all")
    print("wrote", p)


def circle(lon, lat, rad_deg, n=200):
    v0 = hp.ang2vec(lon, lat, lonlat=True)
    a = np.array([0, 0, 1.0]) if abs(v0[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(v0, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v0, e1)
    t = np.linspace(0, 2 * np.pi, n); r = np.radians(rad_deg)
    v = np.cos(r) * v0[:, None] + np.sin(r) * (np.cos(t) * e1[:, None] + np.sin(t) * e2[:, None])
    lo, la = hp.vec2ang(v.T, lonlat=True)
    return lo, la


def load_obs(mp, cut=""):
    return np.load(f"{OUT}/obs_{mp}{cut}.npz"), json.load(open(f"{OUT}/obs_{mp}{cut}.json"))


R = json.load(open(RES)) if RES != "-" and os.path.exists(RES) else None

# ---------------------------------------------------------------- 1 Planck map
if want("map"):
    z, j = load_obs("smica")
    m = z["map"].astype(float).copy(); m[z["mask"] == 0] = hp.UNSEEN
    fig = plt.figure(figsize=(7.2, 4.0))
    hp.mollview(m, fig=fig.number, cmap=PLANCK, min=-300, max=300, title=r"Planck 2018 SMICA, $2^\circ$ FWHM, common mask [$\mu$K]", unit="",
                cbar=True, badcolor="#bbbbbb")
    hp.graticule(dpar=30, dmer=60, alpha=0.35, color="k", lw=0.4)
    save("fig_planck_map")

# ---------------------------------------------------------------- 2 validity
if want("validity"):
    st = S.Setup(PD, cache_dir=OUT)
    fig = plt.figure(figsize=(7.2, 3.0))
    for k, r in enumerate([6, 26]):
        f = st.frac[r].copy()
        hp.mollview(f, fig=fig.number, sub=(1, 2, k + 1), cmap="viridis", min=0, max=1,
                    title=rf"unmasked fraction, $\theta_{{\rm c}}={r}^\circ$ (valid {100*np.mean(f>=S.FMIN):.0f}%)", cbar=True,
                    unit="", notext=True)
    save("fig_validity")

# ---------------------------------------------------------------- 3 templates and filters
if want("templates"):
    st = S.Setup(PD, cache_dir=OUT)
    th = np.radians(np.linspace(0, 40, 801))
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    cm = plt.get_cmap("turbo")
    for n, i in enumerate([1, 4, 6, 8, 10, 12]):
        tc = S.BUB_TC[i]; F = st.kinds["bubble"]["F"][i]
        p = S.profile_from_coeffs(F[0], st.T, th) * 1e-4
        ax[0, 0].plot(np.degrees(th), p, color=cm(n / 5.5), lw=1.4, label=rf"${tc}^\circ$")
    ax[0, 0].set_title(r"collision (linear SW), $z_0=10^{-4}$")
    ax[0, 0].set_ylabel(r"$\Delta T\ [\mu\mathrm{K}]$")
    for n, i in enumerate([0, 2, 4, 5, 7, 8]):
        tc = S.TEX_TC[i]; F = st.kinds["texture"]["F"][i]
        p = S.profile_from_coeffs(F[0], st.T, th) * 1e-4
        ax[0, 1].plot(np.degrees(th), p, color=cm(n / 5.5), lw=1.4, label=rf"${tc}^\circ$")
    ax[0, 1].set_title(r"texture, $\epsilon=10^{-4}$")
    for a in ax[0]:
        a.set_xlabel(r"$\theta$ [deg]"); a.legend(fontsize=7, ncol=2, title=r"$\theta_{\rm c}$", title_fontsize=7)
        a.axhline(0, color="k", lw=0.5)
    L = S.ELL
    for kind, a, idx, tcs in (("bubble", ax[1, 0], [1, 4, 6, 8, 10, 12], S.BUB_TC), ("texture", ax[1, 1], [0, 2, 4, 5, 7, 8], S.TEX_TC)):
        for n, i in enumerate(idx):
            f = st.kinds[kind]["filt"][i][0]
            w = (2 * L + 1) * f * st.kinds[kind]["F"][i][0] * st.T
            w = w / np.max(np.abs(w))
            a.plot(L[2:], w[2:], color=cm(n / 5.5), lw=1.2, label=rf"${tcs[i]}^\circ$")
        a.set_xlabel(r"multipole $\ell$"); a.set_xlim(2, 191); a.axhline(0, color="k", lw=0.5)
        a.set_title(("collision" if kind == "bubble" else "texture") + r": $\ell$-weight of $\Delta\chi^2$")
    ax[1, 0].set_ylabel("normalised weight")
    fig.tight_layout()
    save("fig_templates")

# ---------------------------------------------------------------- 4 dchi2 maps
if want("dchi2"):
    z, j = load_obs("smica")
    fig = plt.figure(figsize=(7.2, 8.6))
    for k, (kind, cmap, vmax, ttl) in enumerate((("sw", "inferno", 22, r"collision (linear SW) $\Delta\chi^2_{\rm SW}$"),
                                                  ("bubble", "magma", 27, r"collision (two-parameter) $\Delta\chi^2_{\rm B}$"),
                                                  ("texture", "plasma", 20, r"texture $\Delta\chi^2_{\rm T}$"))):
        d = z[f"{kind}_max"].astype(float).copy(); d[d <= 0] = hp.UNSEEN
        hp.mollview(d, fig=fig.number, sub=(3, 1, k + 1), cmap=cmap, min=0, max=vmax, title=ttl, badcolor="#9a9a9a",
                    unit="", notext=True)
        hp.graticule(dpar=30, dmer=60, alpha=0.25, color="w", lw=0.3)
        tau = R["stats"]["main"][kind]["tau95"] if R else None
        for q, pk in enumerate(j[kind]["peaks"][:5]):
            lo, la = circle(pk["lon"], pk["lat"], pk["tc"])
            hp.projplot(lo, la, lonlat=True, color="cyan" if q else "lime", lw=1.0)
            hp.projtext(pk["lon"], pk["lat"] + pk["tc"] + 3, str(q + 1), lonlat=True, color="w", fontsize=7)
    save("fig_dchi2_maps")

# ---------------------------------------------------------------- 5 best size map
if want("bestsize"):
    z, j = load_obs("smica")
    fig = plt.figure(figsize=(7.2, 3.0))
    for k, (kind, tcs) in enumerate((("sw", S.BUB_TC), ("texture", S.TEX_TC))):
        a = z[f"{kind}_arg"]; v = np.full(a.size, hp.UNSEEN)
        ok = a >= 0; v[ok] = np.array(tcs)[a[ok]]
        hp.mollview(v, fig=fig.number, sub=(1, 2, k + 1), cmap="turbo", min=min(tcs), max=max(tcs), notext=True,
                    title=("collision" if kind == "sw" else "texture") + r": best-fit $\theta_{\rm c}$ [deg]",
                    unit="", badcolor="#9a9a9a")
    save("fig_bestsize")

# ---------------------------------------------------------------- 6 map comparison
if want("compare"):
    zz = {mp: load_obs(mp) for mp in MAPS}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for a, kind, ttl in zip(ax, ("sw", "bubble", "texture"), ("collision SW", "collision two-par.", "texture")):
        x = zz["smica"][0][f"{kind}_max"].astype(float); ok = x > 0
        for mp in ("nilc", "commander", "sevem"):
            y = zz[mp][0][f"{kind}_max"].astype(float)
            sel = ok & (y > 0)
            idx = np.flatnonzero(sel)[::7]
            a.scatter(x[idx], y[idx], s=1.5, alpha=0.35, color=MCOL[mp], label=MLAB[mp], rasterized=True)
        lim = max(x.max(), 1); a.plot([0, lim], [0, lim], "k-", lw=0.6)
        a.set_title(ttl, fontsize=9); a.set_xlabel(r"SMICA $\Delta\chi^2$")
        a.set_xlim(0, lim); a.set_ylim(0, lim)
    ax[0].set_ylabel(r"other map $\Delta\chi^2$")
    ax[0].legend(fontsize=6, markerscale=5, loc="upper left")
    fig.tight_layout()
    save("fig_compare")

# ---------------------------------------------------------------- 7 null distributions
if want("null") and R:
    fs = sorted(glob.glob(f"{OUT}/null_s*.npz"))
    nl = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("sw", "bubble", "texture")}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    cols = {"sw": "#f4a261", "bubble": "#e9c46a", "texture": "#90be6d"}
    for a, kind, ttl in zip(ax, ("sw", "bubble", "texture"), ("collision SW", "collision two-par.", "texture")):
        x = nl[kind]
        a.hist(x, bins=50, color=cols[kind], edgecolor="#555555", lw=0.3, density=True)
        t = R["stats"]["main"][kind]["tau95"]
        a.axvline(t, color="k", ls="--", lw=1.0, label=r"$\tau_{95}$")
        for mp in MAPS:
            v = R["maps"]["main"][mp][kind]["max"]
            a.axvline(v, color=MCOL[mp], lw=1.4, label=MLAB[mp])
        a.set_title(ttl, fontsize=9); a.set_xlabel(r"$\max\,\Delta\chi^2$ over sky and sizes")
    ax[0].set_ylabel("density"); ax[2].legend(fontsize=6)
    fig.tight_layout()
    save("fig_null")

# ---------------------------------------------------------------- 8 efficiency and bounds
if want("eff") and R and R.get("eff"):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for kind, c, lab in (("sw", "#e63946", "collisions (linear SW)"), ("texture", "#2a9d8f", "textures")):
        rows = R["eff"][kind]["rows"]
        a = np.array([r["amp_muK"] for r in rows]); e = np.array([r["eff"] for r in rows]); ee = np.array([r["eff_err"] for r in rows])
        ax[0].errorbar(a, e, yerr=ee, color=c, marker="o", ms=4, lw=1.3, capsize=2, label=lab)
        nu = np.array([r["N_upper95"] if r["N_upper95"] else np.nan for r in rows])
        ax[1].plot(a, nu, color=c, marker="o", ms=4, lw=1.3, label=lab)
        ax[1].fill_between(a, nu, 1e3, color=c, alpha=0.15)
    ax[0].set_xlabel(r"peak amplitude $|\Delta T_0|$ [$\mu$K]"); ax[0].set_ylabel(r"detection efficiency $\varepsilon$")
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=7)
    ax[1].set_yscale("log"); ax[1].set_ylim(1, 300)
    ax[1].set_xlabel(r"peak amplitude $|\Delta T_0|$ [$\mu$K]"); ax[1].set_ylabel(r"$\bar N$ excluded above (95%)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    save("fig_eff")

if want("effgrid") and R and R.get("eff_grid"):
    g = R["eff_grid"]
    tcs = sorted({r["tc"] for r in g}); amps = sorted({r["amp_muK"] for r in g})
    E = np.full((len(tcs), len(amps)), np.nan)
    for r in g:
        E[tcs.index(r["tc"]), amps.index(r["amp_muK"])] = r["eff"]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    im = ax.imshow(E, origin="lower", cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for i in range(len(tcs)):
        for k in range(len(amps)):
            if np.isfinite(E[i, k]):
                ax.text(k, i, f"{E[i,k]:.2f}", ha="center", va="center", fontsize=7, color="k" if E[i, k] < 0.6 else "w")
    ax.set_xticks(range(len(amps))); ax.set_xticklabels(amps); ax.set_yticks(range(len(tcs))); ax.set_yticklabels(tcs)
    ax.set_xlabel(r"$|\Delta T_0|$ [$\mu$K]"); ax.set_ylabel(r"$\theta_{\rm c}$ [deg]")
    plt.colorbar(im, ax=ax, label=r"$\varepsilon$")
    save("fig_effgrid")

# ---------------------------------------------------------------- 9 injected example
if want("inject"):
    st = S.Setup(PD, cache_dir=OUT)
    rng = np.random.default_rng(11); np.random.seed(11)
    n0 = hp.ang2vec(60.0, 45.0, lonlat=True)
    i = S.BUB_TC.index(12)
    inj = st.inject_map("bubble", i, [-250 / S.TCMB, 0.0], n0)
    m = st.sim_map(rng) + inj
    res = st.scan(st.masked_alm(m))
    fig = plt.figure(figsize=(7.2, 2.6))
    kw = dict(rot=(60, 45), xsize=500, reso=7.0, notext=True)
    mm = m.copy(); mm[~st.good] = hp.UNSEEN
    hp.gnomview(inj, fig=fig.number, sub=(1, 3, 1), cmap=PLANCK, min=-250, max=250, title="injected template", unit=r"$\mu$K", **kw)
    hp.gnomview(mm, fig=fig.number, sub=(1, 3, 2), cmap=PLANCK, min=-250, max=250, title="simulated sky", unit=r"$\mu$K", **kw)
    d = res["sw"]["max"].copy(); d[d <= 0] = hp.UNSEEN
    hp.gnomview(d, fig=fig.number, sub=(1, 3, 3), cmap="inferno", min=0, max=max(30, d.max()), title=r"$\Delta\chi^2_{\rm SW}$", unit="", **kw)
    print("inject example: max near", d[np.degrees(np.arccos(np.clip(st.vec @ n0, -1, 1))) < 6].max(), "global", d.max())
    save("fig_inject_example")

# ---------------------------------------------------------------- 10 3D schematic
if want("schematic"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_subplot(121, projection="3d", computed_zorder=False)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 160), np.linspace(0, np.pi, 80))
    x, y, zc = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    rng = np.random.default_rng(3)
    # smooth random field on the sphere for colour
    alm = hp.synalm(np.r_[0, 0, 1.0 / np.arange(2, 30) ** 2.2], lmax=29)
    val = hp.alm2map(alm, 32, lmax=29)
    pix = hp.vec2pix(32, x, y, zc); col = val[pix]
    caps = [((0.9, 0.15, 0.45), 0.42, +1), ((0.35, 0.9, -0.05), 0.30, -1), ((0.75, 0.35, -0.6), 0.24, +1)]
    for c, r, s in caps:
        c = np.array(c) / np.linalg.norm(c)
        cosd = x * c[0] + y * c[1] + zc * c[2]; th = np.arccos(np.clip(cosd, -1, 1))
        col = col + s * 2.5 * np.std(val) * np.clip((np.cos(th) - np.cos(r)) / (1 - np.cos(r)), 0, None)
    col = (col - col.min()) / (col.max() - col.min())
    ax.plot_surface(x, y, zc, zorder=1, facecolors=PLANCK(col), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    for c, r, s in caps:
        c = np.array(c) / np.linalg.norm(c)
        a = np.array([0, 0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(c, a); e1 /= np.linalg.norm(e1); e2 = np.cross(c, e1)
        t = np.linspace(0, 2 * np.pi, 120)
        ring = 1.01 * (np.cos(r) * c[:, None] + np.sin(r) * (np.cos(t) * e1[:, None] + np.sin(t) * e2[:, None]))
        ax.plot(*ring, color="lime", lw=1.6, zorder=10)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=35)
    ax.set_xlim(-0.72, 0.72); ax.set_ylim(-0.72, 0.72); ax.set_zlim(-0.72, 0.72)
    ax.set_title("sky with collision discs", fontsize=9)
    # right: pipeline as 3D-ish stacked layers
    ax2 = fig.add_subplot(122)
    steps = [("Planck 2018 maps\n(SMICA, NILC, Commander, SEVEM)", "#264653"),
             ("mask, monopole/dipole fit,\ninpainting, 2° smoothing", "#2a9d8f"),
             ("harmonic matched filter\n14 collision + 9 texture sizes", "#e9c46a"),
             (r"global max of $\Delta\chi^2$" + "\nvs 6000 Gaussian skies", "#f4a261"),
             ("injections: efficiency\n" + r"$\Rightarrow$ bound on $\bar N$", "#e76f51")]
    for k, (txt, c) in enumerate(steps):
        y0 = 4 - k
        ax2.add_patch(plt.Rectangle((0.08 + 0.03, y0 - 0.38 - 0.04), 0.84, 0.76, color="#00000033", zorder=1))
        ax2.add_patch(plt.Rectangle((0.08, y0 - 0.38), 0.84, 0.76, color=c, zorder=2))
        ax2.text(0.5, y0, txt, ha="center", va="center", color="w" if k in (0, 1, 4) else "k", fontsize=7.5, zorder=3)
        if k < 4:
            ax2.annotate("", xy=(0.5, y0 - 0.62), xytext=(0.5, y0 - 0.40), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
    ax2.set_xlim(0, 1.05); ax2.set_ylim(-0.6, 4.6); ax2.set_axis_off()
    ax2.set_title("analysis chain", fontsize=9)
    save("fig_schematic")

# ---------------------------------------------------------------- 11 cut comparison
if want("cut") and R and "cut20" in R.get("maps", {}):
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    kinds = ("sw", "bubble", "texture"); xs = np.arange(3)
    for n, mp in enumerate(MAPS):
        for key, mk, off in (("main", "o", -0.12), ("cut20", "s", 0.12)):
            if mp not in R["maps"].get(key, {}):
                continue
            p = [R["maps"][key][mp][k]["p_global"] for k in kinds]
            ax.scatter(xs + off + 0.03 * (n - 1.5), p, color=MCOL[mp], marker=mk, s=22,
                       label=f"{MLAB[mp]}" if key == "main" else None, edgecolor="k", lw=0.3)
    ax.axhline(0.05, color="k", ls="--", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(["collision SW", "collision two-par.", "texture"])
    ax.set_ylabel(r"global $p$-value"); ax.set_ylim(0, 1)
    ax.legend(fontsize=6, ncol=2, title="circles: main mask, squares: |b|>20°", title_fontsize=6)
    save("fig_cut")

# ---------------------------------------------------------------- 12 gallery of strongest features
if want("gallery"):
    z, j = load_obs("smica")
    m = z["map"].astype(float).copy(); m[z["mask"] == 0] = hp.UNSEEN
    sel = [("sw", 0, "SW 1"), ("sw", 1, "SW 2"), ("bubble", 0, "two-amp. 1"),
           ("texture", 0, "texture 1"), ("texture", 1, "texture 2"), ("bubble", 2, "two-amp. 3")]
    plt.rcParams["font.size"] = 8
    fig = plt.figure(figsize=(7.2, 5.6))
    for k, (kind, q, lab) in enumerate(sel):
        pk = j[kind]["peaks"][q]
        hp.gnomview(m, fig=fig.number, sub=(2, 3, k + 1), rot=(pk["lon"], pk["lat"]), reso=6.0, xsize=480,
                    cmap=PLANCK, min=-300, max=300, notext=True, cbar=False,
                    title=f"{lab}, " + r"$\theta_{\rm c}$=" + f"{pk['tc']}°\n" + f"$(l,b)=({pk['lon']:.0f}°, {pk['lat']:+.0f}°)$", badcolor="#9a9a9a")
        lo, la = circle(pk["lon"], pk["lat"], pk["tc"])
        hp.projplot(lo, la, lonlat=True, color="lime", lw=1.3)
    ax = fig.add_axes([0.3, -0.03, 0.4, 0.02])
    import matplotlib as mpl
    mpl.colorbar.ColorbarBase(ax, cmap=PLANCK, norm=mpl.colors.Normalize(-300, 300), orientation="horizontal", label=r"$\Delta T$ [$\mu$K]")
    save("fig_gallery")
