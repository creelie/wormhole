"""Colour figures for v3. Usage: python3 -I figv3.py <sky_dir> <fig_dir> [which...]; csdetect needs CS_SIMDIR."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import cm

SKY, OUT = sys.argv[1], sys.argv[2]
WHICH = sys.argv[3:]
plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm", "savefig.dpi": 220})
BLUE, RED, GOLD, GREEN, PURPLE = "#1f5aa6", "#c0392b", "#e0a800", "#1e8449", "#7d3c98"


def want(n):
    return not WHICH or n in WHICH


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ time curves
if want("timecurves"):
    fig = plt.figure(figsize=(9.2, 4.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    t = np.linspace(0, 2 * np.pi, 400)
    # two unit loops in the plane x-y, tangent at J=(0,0,0)
    x1, y1 = np.cos(t) - 1, np.sin(t)
    x2, y2 = np.cos(t) + 1, np.sin(t)
    # hidden imaginary-time direction drawn along z as a slight lift
    z1 = 0.25 * np.sin(t) ** 2
    z2 = -0.25 * np.sin(t) ** 2
    xx, yy = np.meshgrid(np.linspace(-2.6, 2.6, 2), np.linspace(-1.6, 1.6, 2))
    ax.plot_surface(xx, yy, 0 * xx - 0.6, color="#dfe9f5", alpha=0.6, zorder=0)
    ax.plot(x1, y1, z1, color=BLUE, lw=2.6, zorder=5)
    ax.plot(x2, y2, z2, color=RED, lw=2.6, zorder=5)
    for (xs, ys, zs, c, sgn) in ((x1, y1, z1, BLUE, 1), (x2, y2, z2, RED, -1)):
        for k in (60, 200, 330):
            i, j = (k, k + 6) if sgn > 0 else (k + 6, k)
            ax.quiver(xs[i], ys[i], zs[i], xs[j] - xs[i], ys[j] - ys[i], zs[j] - zs[i], color=c,
                      arrow_length_ratio=2.5, length=1.0, normalize=False, lw=2, zorder=6)
    ax.scatter([0], [0], [0], s=70, color=GOLD, edgecolor="k", zorder=10)
    ax.text(0.05, 0.1, 0.25, r"$J$", fontsize=13, zorder=11)
    ax.text(-1.0, 1.25, 0.35, r"$C^{1}$ ($\alpha$, $+2\pi i$)", color=BLUE, fontsize=10)
    ax.text(0.7, -1.55, -0.5, r"$C^{2}$ ($\beta$, $-2\pi i$)", color=RED, fontsize=10)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((2.4, 1.5, 0.8)); ax.view_init(28, -62); ax.set_axis_off()
    ax.set_title("Opposite time loops joined at $J$", fontsize=10)
    # right: unit circle and poles
    ax2 = fig.add_subplot(1, 2, 2)
    X, Y = np.meshgrid(np.linspace(-2.2, 2.2, 500), np.linspace(-2.2, 2.2, 500))
    Z = X + 1j * Y
    F = np.abs(Z / (3 * Z ** 4 + 10 * Z ** 2 + 3))
    im = ax2.pcolormesh(X, Y, np.log10(F), cmap="magma", vmin=-2, vmax=1, shading="auto")
    ax2.add_patch(Circle((0, 0), 1, fill=False, color="white", lw=2, ls="--"))
    for p, c in ((1 / np.sqrt(3), BLUE), (-1 / np.sqrt(3), RED)):
        ax2.plot(0, p, "o", ms=8, color=c, mec="white")
    for p in (np.sqrt(3), -np.sqrt(3)):
        ax2.plot(0, p, "x", ms=9, color="white", mew=2)
    ax2.text(-1.75, 0.52, r"$+i/\sqrt{3}$", color="white")
    ax2.text(-1.75, -0.66, r"$-i/\sqrt{3}$", color="white")
    ax2.text(0.15, 1.8, r"$\pm i\sqrt{3}$ (outside)", color="white", fontsize=9)
    ax2.text(0.72, -0.95, r"$|\rho|=1$", color="white", fontsize=9)
    ax2.set_aspect("equal"); ax2.set_xlabel(r"Re $\rho$"); ax2.set_ylabel(r"Im $\rho$")
    cb = fig.colorbar(im, ax=ax2, shrink=0.85); cb.set_label(r"$\log_{10}|f(\rho)|$")
    ax2.set_title(r"Poles of $f(\rho)=\rho/(3\rho^4+10\rho^2+3)$", fontsize=10)
    save(fig, "fig_timecurves")

# ------------------------------------------------------------------ entropy
if want("entropy"):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    th = np.linspace(0, np.pi / 2, 400)
    c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        S = -np.nan_to_num(c2 * np.log2(c2)) - np.nan_to_num(s2 * np.log2(s2))
    ax = axs[0]
    ax.fill_between(th / np.pi, 0, S, color="#9ecae1", alpha=0.6)
    ax.plot(th / np.pi, S, color=BLUE, lw=2.2)
    ax.axvline(0.25, color=RED, ls="--"); ax.text(0.26, 0.15, "Bell state\n$S=1$ ebit", color=RED)
    ax.set_xlabel(r"mixing angle $\vartheta/\pi$"); ax.set_ylabel(r"$S(\epsilon_\alpha)$ [ebit]")
    ax.set_title(r"$|\Psi\rangle=\cos\vartheta|01\rangle-\sin\vartheta|10\rangle$", fontsize=10)
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.08)
    ax = axs[1]
    tt = np.linspace(0, 6, 400)
    for g, col in ((0.5, GREEN), (1.0, BLUE), (2.0, PURPLE)):
        thv = np.pi / 4 * (1 - np.exp(-g * tt))
        c2, s2 = np.cos(thv) ** 2, np.sin(thv) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            Sv = -np.nan_to_num(c2 * np.log2(c2)) - np.nan_to_num(s2 * np.log2(s2))
        ax.plot(tt, Sv, color=col, lw=2, label=fr"$\gamma={g}$")
    ax.axhline(1, color="0.4", ls=":")
    ax.set_xlabel(r"time after the split, $t$ [units of $1/\gamma_0$]"); ax.set_ylabel(r"$S$ [ebit]")
    ax.set_title("Entropy of one universe of the pair", fontsize=10); ax.legend(frameon=False)
    fig.tight_layout(); save(fig, "fig_entropy")

# ------------------------------------------------------------------ ricci neck pinch
if want("ricci"):
    fig = plt.figure(figsize=(10, 2.9))
    u = np.linspace(-1, 1, 260); v = np.linspace(0, 2 * np.pi, 120)
    U, V = np.meshgrid(u, v)
    stages = [(0.55, "before: neck forms"), (0.18, "near the singular time"), (0.0, "after surgery: $M_1\\sqcup M_2$")]
    for k, (n, title) in enumerate(stages):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        if n > 0:
            r = np.sqrt(np.clip(1 - U ** 2, 0, None)) * np.sqrt(np.clip((U ** 2 + n ** 2) / (1 + n ** 2), 0, None)) * 1.1
            X = 2.2 * U; Y = r * np.cos(V); Zs = r * np.sin(V)
            neck = np.sqrt((U ** 2 + n ** 2) / (1 + n ** 2))
            curv = 1 / (np.maximum(neck, 0.05))
            cmax = 1 / 0.18
            ax.plot_surface(X, Y, Zs, facecolors=cm.plasma(np.clip((curv - 1) / (cmax - 1), 0, 1)), rstride=2, cstride=2, lw=0, antialiased=True, shade=False)
        else:
            for sgn in (-1, 1):
                uu = np.linspace(0, 1, 140)
                Uc, Vc = np.meshgrid(uu, v)
                r = np.sin(np.pi * Uc) * 0.62
                X = sgn * (0.35 + 1.7 * Uc); Y = r * np.cos(Vc); Zs = r * np.sin(Vc)
                ax.plot_surface(X, Y, Zs, color=BLUE if sgn < 0 else RED, alpha=0.9, rstride=2, cstride=2, lw=0, shade=True)
            ax.text(-1.4, 0, 0.9, r"$M_1$", color=BLUE, fontsize=12)
            ax.text(1.0, 0, 0.9, r"$M_2$", color=RED, fontsize=12)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
        ax.set_box_aspect((2.2, 1.1, 1.1)); ax.view_init(18, -70); ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    m = cm.ScalarMappable(cmap="plasma"); m.set_array([0, 1])
    cax = fig.add_axes([0.35, 0.24, 0.3, 0.035])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal"); cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.set_label(r"curvature $|\mathrm{Rm}|$", fontsize=9)
    save(fig, "fig_ricci")

# ------------------------------------------------------------------ potential
if want("potential"):
    fig = plt.figure(figsize=(9, 3.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    r = np.linspace(0, 3.2, 120); p = np.linspace(0, 2 * np.pi, 120)
    R_, P_ = np.meshgrid(r, p)
    Vv = R_ ** 4 - 10 * R_ ** 2
    ax.plot_surface(R_ * np.cos(P_), R_ * np.sin(P_), Vv, cmap="viridis", rstride=2, cstride=2, lw=0, alpha=0.95)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.sqrt(5) * np.cos(th), np.sqrt(5) * np.sin(th), -25 + 0 * th, color=RED, lw=2.5)
    ax.scatter([0], [0], [0], color=GOLD, s=40, edgecolor="k")
    ax.text(1.5, 2.2, -32, r"$|\varphi|=\sqrt{5}$", color=RED, fontsize=9)
    ax.set_xlabel(r"Re $\varphi$"); ax.set_ylabel(r"Im $\varphi$"); ax.set_zlabel(r"$V$")
    ax.view_init(24, -55); ax.set_title(r"$V(\varphi)=|\varphi|^4-10|\varphi|^2$", fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.linspace(0, 3.3, 300)
    ax2.plot(x, x ** 4 - 10 * x ** 2, color=PURPLE, lw=2.2)
    ax2.axvline(np.sqrt(5), color=RED, ls="--"); ax2.axhline(-25, color="0.5", ls=":")
    ax2.text(np.sqrt(5) + 0.05, 5, r"$\sqrt{5}$", color=RED)
    ax2.text(0.1, -23.5, r"$V_{\min}=-25$", color="0.3")
    ax2.set_xlabel(r"$|\varphi|$"); ax2.set_ylabel(r"$V$"); ax2.set_title("Radial section", fontsize=10)
    fig.tight_layout(); save(fig, "fig_potential")

# ------------------------------------------------------------------ the window
if want("window"):
    R = json.load(open(f"{SKY}/result.json")); SIG = json.load(open(f"{SKY}/out/sigmas.json"))
    st = R["stats"]["main"]
    BUB_TC = np.array([3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 30.])
    TEX_TC = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10.])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, tc, sig, tau, lab in ((axs[0], BUB_TC, np.array(SIG["sw"]), st["sw"]["tau95"], "collision disc (Sachs--Wolfe profile)"),
                                  (axs[1], TEX_TC, np.array(SIG["tex"]), st["texture"]["tau95"], "cosmic texture")):
        a50 = np.sqrt(tau) * sig; a95 = (np.sqrt(tau) + 1.6449) * sig
        lo, hi = tc.min(), tc.max()
        ax.fill_between(tc, a95, 2000, color="#f4a6a6", alpha=0.85, lw=0)
        ax.fill_between(tc, a50, a95, color="#fbd9a5", alpha=0.9, lw=0)
        ax.fill_between(tc, 20, a50, color="#bfe3c4", alpha=0.9, lw=0)
        ax.plot(tc, a50, color=RED, lw=2); ax.plot(tc, a95, color="#8e1b10", lw=1.4, ls="--")
        ax.axvspan(0.5 if lo < 2 else 1.5, lo, color="0.85", hatch="//", alpha=0.6, lw=0)
        ax.axvspan(hi, 60 if hi > 20 else 20, color="0.85", hatch="//", alpha=0.6, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(40, 1500)
        ax.set_xlim(1.5, 60) if hi > 20 else ax.set_xlim(0.6, 20)
        ax.set_xlabel(r"angular radius $\theta_c$ [deg]"); ax.set_title(lab, fontsize=10)
        from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
        tk = [2, 3, 5, 10, 20, 30, 50] if hi > 20 else [1, 2, 3, 5, 10, 20]
        ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in tk]))
        ax.xaxis.set_minor_locator(NullLocator())
        yt = [50, 100, 200, 500, 1000]
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in yt]))
        ax.text(0.04, 0.93, "closed: seen anywhere on\nthe valid sky with $\\geq$95% prob.", transform=ax.transAxes, fontsize=8, va="top", color="#7b1d14")
        ax.text(0.04, 0.10, "open window: below the\nglobal detection edge", transform=ax.transAxes, fontsize=8, color="#145a32")
    # data points
    ax = axs[0]
    ax.errorbar([10], [144], yerr=[32], fmt="*", ms=14, color=BLUE, mec="k", capsize=3, label="Cold Spot (blind SW fit)")
    ax.errorbar([4], [175], yerr=[38], fmt="o", ms=7, color=PURPLE, mec="k", capsize=3, label=r"south-pole disc $(268^\circ,-77^\circ)$")
    ax.text(2.0, 1250, "unscanned", fontsize=8, color="0.3"); ax.text(32, 1250, "unscanned", fontsize=8, color="0.3")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.set_ylabel(r"central temperature $|\Delta T_0|$ [$\mu$K]")
    ax = axs[1]
    ax.errorbar([6], [215], yerr=[50], fmt="*", ms=14, color=BLUE, mec="k", capsize=3, label="Cold Spot (Planck, this work)")
    ax.errorbar([4.9], [7.3e-5 * 2.7255e6], yerr=[[3.6e-5 * 2.7255e6], [2.5e-5 * 2.7255e6]], fmt="D", ms=6, color=GOLD, mec="k", capsize=3, label="Cold Spot (WMAP, Cruz et al. 2007)")
    ax.text(0.65, 1250, "unscanned", fontsize=8, color="0.3"); ax.text(10.6, 1250, "unscanned", fontsize=8, color="0.3")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout(); save(fig, "fig_window")
    for tc, s in zip(BUB_TC, SIG["sw"]):
        pass
    print("SW a50", dict(zip(BUB_TC.tolist(), np.round(np.sqrt(st['sw']['tau95']) * np.array(SIG['sw'])).tolist())))
    print("TX a50", dict(zip(TEX_TC.tolist(), np.round(np.sqrt(st['texture']['tau95']) * np.array(SIG['tex'])).tolist())))
    print("SW a95", np.round((np.sqrt(st['sw']['tau95']) + 1.6449) * np.array(SIG['sw'])).tolist())
    print("TX a95", np.round((np.sqrt(st['texture']['tau95']) + 1.6449) * np.array(SIG['tex'])).tolist())


# ------------------------------------------------------------------ Cold Spot imprint simulations
if want("csdetect"):
    import glob, os
    from scipy.stats import norm
    SIM = os.environ["CS_SIMDIR"]  # directory with the cs_<amp>.json outputs of sim_cs.py
    rows = []
    for f in glob.glob(f"{SIM}/cs_*.json"):
        d = json.load(open(f)); r = d["rec"]; v = [x for x in r if x["valid"]]
        rows.append((d["amp"], np.mean([x["det"] for x in r]), len(r), np.mean([x["det"] for x in v]), len(v), d))
    rows.sort()
    A = np.array([r[0] for r in rows]); pa = np.array([r[1] for r in rows]); na = np.array([r[2] for r in rows])
    pv = np.array([r[3] for r in rows]); nv = np.array([r[4] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axs[0]
    ax.axvspan(165, 265, color=GOLD, alpha=0.18, lw=0)
    ax.axvline(215, color=GOLD, lw=1.4)
    ax.text(217, 0.04, "Cold Spot\n$215\\pm50\\,\\mu$K", color="#8a6d00", fontsize=8.5)
    ax.axvline(246, color=RED, lw=1.2, ls="-"); ax.text(249, 0.86, "edge", color=RED, fontsize=8.5)
    ax.axvline(328, color=RED, lw=1.2, ls="--"); ax.text(331, 0.86, "closure", color=RED, fontsize=8.5)
    aa = np.linspace(80, 520, 300)
    ax.plot(aa, norm.cdf(aa / 49.6 - np.sqrt(24.6)), color="0.45", lw=1.3, ls=":", label="edge bound (Prop.)")
    ax.errorbar(A, pv, yerr=np.sqrt(pv * (1 - pv) / nv), fmt="o-", color=BLUE, ms=4.5, lw=1.6, capsize=2, label="valid centre")
    ax.errorbar(A, pa, yerr=np.sqrt(pa * (1 - pa) / na), fmt="s-", color=PURPLE, ms=4, lw=1.3, capsize=2, label="any centre")
    ax.set_xlabel("central amplitude $|\\Delta T_0|$ ($\\mu$K)"); ax.set_ylabel("probability of detection")
    ax.set_ylim(0, 1.02); ax.set_xlim(80, 520); ax.legend(loc="center right", fontsize=8.5, frameon=False)
    ax.set_title("(a) blind detection of a $6^\\circ$ texture", fontsize=10)
    ax = axs[1]
    cols = {100.0: GREEN, 215.0: GOLD, 320.0: RED}
    bins = np.linspace(0, 80, 41)
    for a, c in cols.items():
        r = [x for x in rows if x[0] == a]
        if not r: continue
        dm = [x["dmax_near"] for x in r[0][5]["rec"] if x["valid"]]
        ax.hist(dm, bins=bins, histtype="stepfilled", alpha=0.35, color=c)
        ax.hist(dm, bins=bins, histtype="step", lw=1.5, color=c, label=f"{a:.0f} $\\mu$K")
    ax.axvline(24.6, color="k", lw=1.2, ls="--"); ax.text(25.5, ax.get_ylim()[1] * 0.92, "$\\tau_{95}=24.6$", fontsize=8.5)
    ax.set_xlabel("$\\max D_{\\rm T}$ within $3^\\circ$ of the true centre"); ax.set_ylabel("simulations (valid centres)")
    ax.legend(fontsize=8.5, frameon=False, title="injected", title_fontsize=8.5)
    ax.set_title("(b) statistic at the imprint", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_csdetect")
