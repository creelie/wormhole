import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, skylib as S
L = S.LMAX
def intP(c, lmax):
    P = S.legendre_table(np.array([c]), lmax + 2)[:, 0]
    I = np.empty(lmax + 2); I[0] = 1 - c
    for l in range(1, lmax + 2):
        I[l] = (P[l - 1] - P[l + 1]) / (2 * l + 1)
    return I
worst = 0
for tc in S.BUB_TC:
    c = np.cos(np.radians(tc)); I = intP(c, L)
    IxP = np.empty(L + 1); IxP[0] = (1 - c * c) / 2
    for l in range(1, L + 1):
        IxP[l] = ((l + 1) * I[l + 1] + l * I[l - 1]) / (2 * l + 1)
    exact = 2 * np.pi * S.TCMB / (1 - c) * (IxP - c * I[:L + 1])
    q = S.bubble_coeffs(tc)[0]
    worst = max(worst, np.max(np.abs(q - exact)) / np.max(np.abs(exact)))
wt = 0
for tc in S.TEX_TC:
    a = S.texture_coeffs(tc)[0]
    tcr = np.radians(tc); xs = np.cos(np.sqrt(3) / 2 * tcr)
    def g(x):
        t = np.arccos(np.clip(x, -1, 1))
        return S.TCMB * np.where(t <= np.sqrt(3)/2*tcr, 1/np.sqrt(1+4*t**2/tcr**2), 0.5*np.exp(-(t**2-(np.sqrt(3)/2*tcr)**2)/(2*tcr**2)))
    b = S.zonal_coeffs(g, [(xs, 1.0, 1200), (-1.0, xs, 6000)])
    wt = max(wt, np.max(np.abs(a - b)) / np.max(np.abs(b)))
print(json.dumps({"bubble_rel": worst, "texture_rel": wt}))
