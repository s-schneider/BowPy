"""
Testscript for all cases
"""
import os

noise = np.fromfile('../data/test_datasets/randnumbers/noisearr.txt')
noise = noise.reshape(20,300)

with open("../data/test_datasets/ricker/rickerlist.dat", 'r') as fh:
	rickerlist = np.array(fh.read().split()).astype('str')

noisefoldlist = ["no_noise","10pct_noise", "20pct_noise", "50pct_noise", "60pct_noise", "80pct_noise"]
noiselevellist = np.array([0., 0.1, 0.2, 0.5, 0.6, 0.8]) 

peaks = np.array([[-13.95      ,   6.06      ,  20.07      ],[  8.46648822,   8.42680793,   8.23354933]])
errors = []
#stream = read_st(PATH)

for i, noisefolder in enumerate(noisefoldlist):
	print("##################### NOISELVL %i %% #####################\n" % int(noiselevellist[i] * 100.) )
	for filein in rickerlist:
		print("##################### CURRENT FILE %s  #####################\n" % filein )
		PICPATH = filein[:filein.rfind("/"):] + "/" + noisefolder + "/"
		if not os.path.isdir(PICPATH):
			os.mkdir(PICPATH)
		PATH = filein
		stream = read_st(PATH)
		#if i != 0:
		data = stream2array(stream.copy(), normalize=True) + noiselevellist[i] * noise
		srs = array2stream(data)

			
		name = 'boxcar_auto_noise_' + str(noiselevellist[i]) +  '.png'
		picpath = PICPATH + name
		st_rec = fk_reconstruct(srs, slopes=[-2,2], deltaslope=0.001, maskshape=['boxcar', None], solver='cg',method='interpolate', mu=42, tol=1e-12, peakinput=peaks)
		st_rec.normalize()
		fku.plot_data(stream2array(st_rec), savefig=picpath)


		name = 'boxcar_size1_noise_' + str(noiselevellist[i]) +  '.png'
		picpath = PICPATH + name
		st_rec = fk_reconstruct(srs, slopes=[-2,2], deltaslope=0.001, maskshape=['boxcar', None], solver='cg',method='interpolate', mu=42, tol=1e-12, peakinput=peaks)
		st_rec.normalize()
		fku.plot_data(stream2array(st_rec), savefig=picpath)


		taperrange = [0.5, 1, 1.5]
		for ts in taperrange:
			print("##################### %s, NOISE: %f, :CURRENTLY TAPERING WITH %f  #####################\n" % (filein, int(noiselevellist[i] * 100.), ts) )
			try:
				st_rec = fk_reconstruct(srs, slopes=[-2,2], deltaslope=0.001, maskshape=['taper', ts], solver='cg',method='interpolate', mu=42, tol=1e-12, peakinput=peaks)
				name = 'taper_' + str(ts) + "_" + str(noiselevellist[i]) +  '.png'
				picpath = PICPATH + name
				st_rec.normalize()
				fku.plot_data(stream2array(st_rec), savefig=picpath)
			except:
				error.append(picpath)
				continue

		bwrange = [1,2,4,8]
		for bw in bwrange:
			print("##################### %s, NOISE: %f, :CURRENTLY BUTTERWORTHING WITH %f  #####################\n" % (filein, int(noiselevellist[i] * 100.), bw) )
			try:
				st_rec = fk_reconstruct(srs, slopes=[-2,2], deltaslope=0.001, maskshape=['butterworth', bw], solver='cg',method='interpolate', mu=42, tol=1e-12, peakinput=peaks)
				name = 'butterworth_' + str(bw) + "_" + str(noiselevellist[i]) +  '.png'
				picpath = PICPATH + name
				st_rec.normalize()
				fku.plot_data(stream2array(st_rec), savefig=picpath)
			except:
				error.append(picpath)
				continue
