#!/usr/bin/env bash
# Download Commander and SEVEM one at a time, reduce to alm, record checksum, delete the full map.
set -euo pipefail
PD="$1"; DD="$2"; CODE="$3"
U=https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb
for n in commander sevem; do
  f="COM_CMB_IQU-${n}_2048_R3.00_full.fits"
  for i in 1 2 3 4; do curl -sS --retry 3 -o "$PD/$f" "$U/$f" && break; sleep $((i*10)); done
  (cd "$PD" && sha256sum "$f" >> SHA256SUMS)
  python3 -I "$CODE/prep.py" "$PD" "$DD" "$n"
  rm -f "$PD/$f"
done
echo FETCH_DONE
