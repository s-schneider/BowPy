
st = read_st('data/test_datasets/ricker/results/mp_coda.pickle')

from sipy.util.fkutil import line_set_zero

# filterlist = [ ['spike', None, None ], 
# 				['boxcar', 0, 1 ], ['boxcar', 0, 2 ], ['boxcar', 0, 3 ], ['boxcar', 0, 4 ], 
filterlist = [	['taper', 22, 2 ], ['boxcar', 37, 2 ], ]
				# ['taper', 3, 2 ], ['taper', 4, 2 ],
				# ['taper', 1, 3 ], ['taper', 2, 3 ], ['taper', 3, 3 ], ['taper', 4, 3 ],
				# ['taper', 1, 4 ], ['taper', 2, 4 ], ['taper', 3, 4 ], ['taper', 4, 4 ],
				# ['butterworth', 1, 2 ], ['butterworth', 2, 2 ], ['butterworth', 4, 2 ], ['butterworth', 8, 2 ],
				# ['butterworth', 1, 3 ], ['butterworth', 2, 3 ], ['butterworth', 4, 3 ], ['butterworth', 8, 3 ],
				# ['butterworth', 1, 4 ], ['butterworth', 2, 4 ], ['butterworth', 4, 4 ], ['butterworth', 8, 4 ]]



for i, shape in enumerate(filterlist):
	print("%i of %i  done" % (i+1, len(filterlist)))#, end="\r")
	st_tmp    = st.copy()
	stfk      = fku.fktrafo(st_tmp)
	fkfiltered= line_set_zero(stfk, shape)
	stfiltered= fku.ifktrafo(fkfiltered, st_tmp)
	streamfk= fk_filter(st, ftype='eliminate', fshape = shape)
	stfk = fk_filter(st_tmp, fshape = shape, ftype='eliminate')
	pname     = 'mp_coda' + str(shape[0]) + '_' + str(shape[1]) + '_' +  str(shape[2]) + '.png'
	vname	  = 'vespa_' + pname
	# plogname  = 'log' + pname
	# streamname= 'seismogram' + pname
	# fku.plotfk(fkfiltered, logscale=False, savefig=pname)
	# fku.plotfk(fkfiltered, logscale=True, savefig=plogname)
	# plot(stfiltered, savefig=streamname)
	null = vespagram(stfk, -5, 10, 0.05, plot='classic', markphases=False, savefig=vname)