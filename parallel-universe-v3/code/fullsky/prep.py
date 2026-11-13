"""Reduce a Planck PR3 CMB map to harmonic coefficients (lmax 383, muK, native beam).
Usage: python3 -I prep.py <planck_dir> <data_dir> <name> [<name> ...]"""
import sys, numpy as np, healpy as hp
PD, DD = sys.argv[1], sys.argv[2]
for name in sys.argv[3:]:
    m = hp.read_map(f"{PD}/COM_CMB_IQU-{name}_2048_R3.00_full.fits", field=0) * 1e6
    alm = hp.map2alm(m, lmax=383, use_pixel_weights=True)
    hp.write_alm(f"{DD}/alm_{name}_l383.fits", alm, overwrite=True)
    print(name, "done", np.std(m))
