import sys
drange = np.linspace(0,180,361)
fs = 22
xlabel = 'Distance (degrees)'
ylabel = 'Time (minutes)'
pname = ['PP', 'P^410P', 'P^660P', 'SS', 'S^410S', 'S^660S']
plist = []
for name in pname:
	phase = []
	for distance in drange:
		print("current distance: %f " % distance, end="\r")
		sys.stdout.flush()	
		PP = m.get_travel_times(50, distance, [name])
		#if len(PP) > 1:
		#	index = min(len(PP))-1
		#	phase.append(PP[index].time)
		#else:
		try:
			phase.append(PP[0].time/60.)
		except:
			phase.append(0)

	plist.append(phase)

fig, ax = plt.subplots()
for i, phase in enumerate(plist):
	ax.plot(drange, phase, label=pname[i], linewidth=4)

ax.set_xlabel(xlabel, fontsize=fs)
ax.set_ylabel(ylabel, fontsize=fs)
ax.tick_params(axis='both', which='major', labelsize=fs)
#ax.legend(loc=0, fontsize=fs)
#ax.legend(bbox_to_anchor=(1.2, 1))
fig.set_size_inches(18,24)
fig.savefig('traveltime_no4.png', dpi=300)

