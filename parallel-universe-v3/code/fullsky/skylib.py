"""Full-sky matched-filter search for bubble-collision discs and cosmic textures.

Conventions
-----------
* Maps at HEALPix Nside 64, band limit lmax = 191, smoothed to 2 deg FWHM; units muK.
* Total transfer T_l = B_l(5') W_l(2048) G_l(2 deg); signal S_l = C_l T_l^2 + N_l.
* A zonal template f(theta) (per unit dimensionless amplitude, in muK) has Legendre
  coefficients f_l = 2 pi int f P_l(cos theta) sin theta d theta.
* Full-sky generalised least squares in harmonic space:
      b_k(n0)  = sum_lm f_kl T_l / S_l  d_lm Y_lm(n0)
      A_kk'    = sum_l (2l+1)/(4 pi) f_kl f_k'l T_l^2 / S_l
      dchi2    = b^T A^-1 b,  a_hat = A^-1 b.
  On the masked sky the same filter is applied to the masked, monopole/dipole-cleaned map;
  the statistic is then calibrated with simulations processed identically.
"""
import numpy as np
import healpy as hp

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
LMAX = 3 * NSIDE - 1
TCMB = 2.7255e6
SIGMA_N = 1.0                                   # muK white noise per pixel
ELL = np.arange(LMAX + 1)
BUB_TC = [3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 30]
TEX_TC = [1, 2, 3, 4, 5, 6, 7, 8, 10]
MARGIN = 2.0                                    # deg added to the disc for the validity test
FMIN = 0.85                                     # minimum unmasked fraction inside disc+margin


def legendre_table(x, lmax=LMAX):
    P = np.empty((lmax + 1, x.size))
    P[0] = 1.0
    if lmax:
        P[1] = x
    for l in range(2, lmax + 1):
        P[l] = ((2 * l - 1) * x * P[l - 1] - (l - 1) * P[l - 2]) / l
    return P


def gl_nodes(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def zonal_coeffs(f, pieces):
    """f_l = 2 pi int_{-1}^{1} f(x) P_l(x) dx with Gauss-Legendre on the given x-pieces."""
    out = np.zeros(LMAX + 1)
    for a, b, n in pieces:
        x, w = gl_nodes(a, b, n)
        out += 2 * np.pi * legendre_table(x) @ (w * f(x))
    return out


def bubble_coeffs(tc_deg):
    """Two components: w0 = (cos t - cos tc)/(1 - cos tc), w1 = 1 - w0, inside the disc (x > cos tc)."""
    ct = np.cos(np.radians(tc_deg))
    w0 = lambda x: TCMB * (x - ct) / (1 - ct)
    w1 = lambda x: TCMB * (1 - x) / (1 - ct)
    pc = [(ct, 1.0, 600)]
    return np.array([zonal_coeffs(w0, pc), zonal_coeffs(w1, pc)])


def texture_coeffs(tc_deg):
    tc = np.radians(tc_deg); ts = np.sqrt(3) / 2 * tc; xs = np.cos(ts)
    def g(x):
        t = np.arccos(np.clip(x, -1, 1))
        inner = 1 / np.sqrt(1 + 4 * t ** 2 / tc ** 2)
        outer = 0.5 * np.exp(-(t ** 2 - ts ** 2) / (2 * tc ** 2))
        return TCMB * np.where(t <= ts, inner, outer)
    return np.array([zonal_coeffs(g, [(xs, 1.0, 600), (-1.0, xs, 3000)])])


def profile_from_coeffs(fl, Tl, theta):
    """Smoothed radial profile sum (2l+1)/(4pi) f_l T_l P_l(cos theta)."""
    P = legendre_table(np.cos(theta))
    return ((2 * ELL + 1) / (4 * np.pi) * fl * Tl) @ P


class Setup:
    def __init__(self, planck_dir, mask_path=None, extra_cut_deg=0.0, cache_dir=None):
        self.cache = None if cache_dir is None else f"{cache_dir}/validity_cut{int(extra_cut_deg)}.npz"
        th = np.loadtxt(f"{planck_dir}/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt")
        self.cl = np.zeros(LMAX + 1)
        L = th[:, 0].astype(int); s = (L >= 2) & (L <= LMAX)
        self.cl[L[s]] = 2 * np.pi * th[s, 1] / (L[s] * (L[s] + 1))
        self.G = hp.gauss_beam(np.radians(2.0), LMAX)
        self.T = hp.gauss_beam(np.radians(5 / 60), LMAX) * hp.pixwin(2048)[: LMAX + 1] * self.G
        self.N = SIGMA_N ** 2 * 4 * np.pi / NPIX
        self.S = self.cl * self.T ** 2 + self.N
        mhi = hp.read_map(mask_path or f"{planck_dir}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")
        self.mask = (hp.ud_grade(mhi, NSIDE) > 0.9).astype(float)
        if extra_cut_deg > 0:
            _, b = hp.pix2ang(NSIDE, np.arange(NPIX), lonlat=True)
            self.mask[np.abs(b) < extra_cut_deg] = 0.0
        self.good = self.mask > 0
        self.vec = np.array(hp.pix2vec(NSIDE, np.arange(NPIX))).T
        self.kinds = {}
        for name, tcs, fn in (("bubble", BUB_TC, bubble_coeffs), ("texture", TEX_TC, texture_coeffs)):
            F = [fn(tc) for tc in tcs]
            keep = (ELL >= 2).astype(float)            # monopole and dipole are removed from the data
            filt = [f * self.T / self.S * keep for f in F]
            A = [np.einsum("l,il,jl->ij", keep * (2 * ELL + 1) / (4 * np.pi) * self.T ** 2 / self.S, f, f) for f in F]
            self.kinds[name] = dict(tc=tcs, F=F, filt=filt, A=A, Ainv=[np.linalg.inv(a) for a in A])
        self.valid = self._validity()

    def _validity(self):
        """valid[kind][i] : boolean map of centres whose disc (radius tc+margin) is >= FMIN unmasked."""
        radii = sorted({tc for k in self.kinds.values() for tc in k["tc"]})
        import os
        if self.cache and os.path.exists(self.cache):
            z = np.load(self.cache)
            self.frac = {r: z[f"r{r}"] for r in radii}
            return {name: [self.frac[tc] >= FMIN for tc in k["tc"]] for name, k in self.kinds.items()}
        cosr = {r: np.cos(np.radians(r + MARGIN)) for r in radii}
        frac = {r: np.zeros(NPIX) for r in radii}
        v32 = self.vec.astype(np.float32); g32 = self.mask.astype(np.float32)
        for i0 in range(0, NPIX, 2048):
            c = v32[i0:i0 + 2048] @ v32.T
            for r in radii:
                inside = c >= cosr[r]
                frac[r][i0:i0 + 2048] = (inside @ g32) / inside.sum(1)
        self.frac = frac
        if self.cache:
            np.savez(self.cache, **{f"r{r}": frac[r] for r in radii})
        return {name: [frac[tc] >= FMIN for tc in k["tc"]] for name, k in self.kinds.items()}

    # ---------------- maps ----------------
    def data_map(self, alm383_path):
        alm = hp.read_alm(alm383_path)
        lmax_in = hp.Alm.getlmax(alm.size)
        a = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
        for m in range(LMAX + 1):
            for_l = np.arange(m, LMAX + 1)
            a[hp.Alm.getidx(LMAX, for_l, m)] = alm[hp.Alm.getidx(lmax_in, for_l, m)]
        fl = self.G.copy(); fl[:2] = 0
        return hp.alm2map(hp.almxfl(a, fl), NSIDE, lmax=LMAX)

    def sim_map(self, rng):
        alm = hp.synalm(self.cl * self.T ** 2, lmax=LMAX, new=True)
        m = hp.alm2map(alm, NSIDE, lmax=LMAX)
        return m + SIGMA_N * rng.standard_normal(NPIX)

    def inpaint(self, m, niter=400):
        """Diffusive inpainting: masked pixels repeatedly take the mean of their neighbours."""
        if not hasattr(self, "_nb"):
            nb = hp.get_all_neighbours(NSIDE, np.arange(NPIX))
            self._bad = np.flatnonzero(~self.good)
            nbb = nb[:, self._bad]
            self._nbb = np.where(nbb < 0, self._bad[None, :], nbb)
            self._nb = True
        x = m.copy(); x[self._bad] = 0.0
        for _ in range(niter):
            x[self._bad] = x[self._nbb].mean(0)
        return x

    def masked_alm(self, m):
        """Remove monopole and dipole fitted on the unmasked sky, inpaint the mask, transform."""
        good = self.good
        X = np.column_stack([np.ones(good.sum()), self.vec[good]])
        coef, *_ = np.linalg.lstsq(X, m[good], rcond=None)
        mm = m - (coef[0] + self.vec @ coef[1:])
        mm = self.inpaint(mm)
        return hp.map2alm(mm, lmax=LMAX, iter=1)

    # ---------------- statistics ----------------
    def scan(self, alm, keep_maps=False):
        """For each kind: per-size dchi2 maps restricted to valid centres; max over sizes.
        Bubble also gets the one-amplitude linear Sachs-Wolfe statistic 'sw' (component w0)."""
        out = {}
        for name, k in self.kinds.items():
            best = np.zeros(NPIX); arg = np.full(NPIX, -1)
            if name == "bubble":
                best_sw = np.zeros(NPIX); arg_sw = np.full(NPIX, -1)
            per = []
            for i, (filt, Ainv, A) in enumerate(zip(k["filt"], k["Ainv"], k["A"])):
                b = np.array([hp.alm2map(hp.almxfl(alm, f), NSIDE, lmax=LMAX) for f in filt])
                d = np.einsum("in,ij,jn->n", b, Ainv, b)
                d[~self.valid[name][i]] = 0.0
                upd = d > best; best[upd] = d[upd]; arg[upd] = i
                if name == "bubble":
                    dsw = b[0] ** 2 / A[0, 0]; dsw[~self.valid[name][i]] = 0.0
                    u2 = dsw > best_sw; best_sw[u2] = dsw[u2]; arg_sw[u2] = i
                if keep_maps:
                    per.append((d, Ainv @ b))
            out[name] = dict(max=best, arg=arg)
            if name == "bubble":
                out["sw"] = dict(max=best_sw, arg=arg_sw)
            if keep_maps:
                out[name]["per"] = per
        return out

    def inject_map(self, kind, i_size, amps, n0):
        """Smoothed template of the given kind/size with amplitudes amps centred at unit vector n0."""
        F = self.kinds[kind]["F"][i_size]
        theta = np.linspace(0, np.pi, 6001)
        prof = sum(a * profile_from_coeffs(f, self.T, theta) for a, f in zip(amps, F))
        ang = np.arccos(np.clip(self.vec @ n0, -1, 1))
        return np.interp(ang, theta, prof)


def peaks(stat, good, thresh, sep_deg=5.0):
    """Local maxima of a statistic map above thresh, greedily separated by sep_deg."""
    idx = np.flatnonzero((stat >= thresh) & good)
    idx = idx[np.argsort(stat[idx])[::-1]]
    chosen = []
    for p in idx:
        v = np.array(hp.pix2vec(NSIDE, p))
        if all(np.degrees(np.arccos(np.clip(v @ np.array(hp.pix2vec(NSIDE, q)), -1, 1))) > sep_deg for q in chosen):
            chosen.append(p)
    return chosen
