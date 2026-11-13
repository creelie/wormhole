import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, healpy as hp, skylib as S
PD, DD, OUT = sys.argv[1:4]
st = S.Setup(PD, cache_dir=OUT)
res = {}
for mp in ["smica"]:
    alm = st.masked_alm(st.data_map(f"{DD}/alm_{mp}_l383.fits"))
    r = st.scan(alm, keep_maps=True)
    j = json.load(open(f"{OUT}/obs_{mp}.json"))
    out = {}
    for kind, kk, tcs in (("sw", "bubble", S.BUB_TC), ("bubble", "bubble", S.BUB_TC), ("texture", "texture", S.TEX_TC)):
        rows = []
        for pk in j[kind]["peaks"][:6]:
            p = hp.ang2pix(S.NSIDE, pk["lon"], pk["lat"], lonlat=True); i = tcs.index(pk["tc"])
            A = st.kinds[kk]["A"][i]; b_amp = r[kk]["per"][i][1][:, p]
            if kind == "sw":
                # one-amplitude fit: a = b0/A00 ; recover b0 = (A @ ahat)[0]
                b0 = (A @ b_amp)[0]; a = b0 / A[0, 0]; e = 1 / np.sqrt(A[0, 0]); rows.append([a * S.TCMB, e * S.TCMB])
            else:
                e = np.sqrt(np.diag(np.linalg.inv(A))); rows.append([list(b_amp * S.TCMB), list(e * S.TCMB)])
        out[kind] = rows
    res[mp] = out
json.dump(res, open(f"{OUT}/peakamps.json", "w"), indent=1, default=float)
print(json.dumps(res, default=float)[:1500])
