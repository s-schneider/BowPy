from gatspy import datasets, periodic
import numpy as np

rrlyrae = datasets.fetch_rrlyrae()

lcid = rrlyrae.ids[0]

t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
mask = (filts == 'r')
t_r, mag_r, dmag_r = t[mask], mag[mask], dmag[mask]