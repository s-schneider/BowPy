#get data with instaseis
import obspy
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees, locations2degrees
from obspy import read as read_st
from obspy import read_inventory as read_inv
from obspy import read_events as read_cat
from obspy.taup import TauPyModel

import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import scipy.signal as signal
from numpy import genfromtxt

from sipy.util.array_util import get_coords

import os
import datetime
import sipy.filter.fk as fk
from sipy.filter.fk import fk_filter
import sipy.util.fkutil as fku
import instaseis as ins

uniform=False
real=True
# db = ins.open_db("/Users/Simon/dev/instaseis/10s_PREM_ANI_FORCES")
#db = ins.open_db("/local/s_schn42/instaseis/10s_PREM_ANI_FORCES")
db = ins.open_db("/Volumes/UNTITLED/10s_PREM_ANI_FORCES")
tofe = obspy.UTCDateTime(2009, 10, 24, 14, 40, 44, 770000)
lat = -6.1165
lon = 130.429
depth = 140300
aperture=20
no_of_stations=20
# in degrees
distance_to_source=100
# magnitude of randomness 
magn=1.

#output

streamfile = "synth1.pickle"

qstfile = None #"SYNTH_OUT.QST"
invfile = "synth1_inv.xml"
catfile = "synth1_cat.xml"

source = ins.Source(
latitude=lat, longitude=lon, depth_in_m=depth,
m_rr = 0.526e26 / 1E7,
m_tt = -2.1e26 / 1E7,
m_pp = -1.58e26 / 1E7,
m_rt = 1.08e+26 / 1E7,
m_rp = 2.05e+26 / 1E7,
m_tp = 0.607e+26 / 1E7,
origin_time=tofe
)

x = []
station_range = np.linspace(0,aperture-1,no_of_stations) + 100.
#r = np.random.randn(no_of_stations)
#randrange = station_range + magn * r  
#randrange[0] = station_range[0]
#randrange[no_of_stations-1] = station_range[no_of_stations-1]
# while randrange.max() > randrange[19]:
# 	i = randrange.argmax()
# 	randrange[i] = randrange[i]-0.1

# randrange.sort()


randrange = np.array([ 100.        ,  101.74222711,  102.8608334 ,  104.13732881,
        105.28349288,  106.78556465,  107.488736  ,  108.34593815,
        109.6161234 ,  110.27633321,  111.35174204,  112.90012348,
        113.63875348,  114.34439107,  115.29740496,  116.96181391,
        117.24875298,  117.77155468,  118.14675172,  119.        ])




with open( qstfile, "w") as fh:
	if uniform:
		k=0
		for i in station_range:
			slon = i
			name="X"+str(int(k))
			print(name)
			x.append(ins.Receiver(latitude="54", longitude=str(slon), network="LA", station=name ))
			latdiff = gps2dist_azimuth(54,0,54,slon)[0]/1000.
			fh.write("%s    lat:     54.0 lon:     %f elevation:   0.0000 array:LA  xrel:      %f yrel:      0.00 name:ADDED BY SIMON \n" % (name, slon, latdiff))
			k+=1
	elif real:

		for station in network:
			x.append(ins.Receiver(latitude=str(station.latitude), longitude=str(station.latitude), network=str(network.code), station=str(station.code) ))
	else:
		for i, slon in enumerate(randrange):
			name="X"+str(int(i))
			x.append(ins.Receiver(latitude="54", longitude=slon, network="RA", station=name ))
			latdiff = gps2dist_azimuth(54,0,54,slon)[0]/1000.
			fh.write("%s    lat:     54.0 lon:     %f elevation:   0.0000 array:RA  xrel:      %f yrel:      0.00 name:ADDED BY SIMON \n" % (name, slon, latdiff))		

st_synth = []    
for i in range(len(x)):
    st_synth.append(db.get_seismograms(source=source, receiver=x[i]))



stream=st_synth[0]
for i in range(len(st_synth))[1:]:
    stream.append(st_synth[i][0])


#stream.write("../data/synth.sac", format="SAC")
#stream.write("../data/SYNTH.QHD", format="Q")


"""
Write quakeml file
"""


with open( catfile, "w") as fh:
	fh.write("<?xml version=\'1.0\' encoding=\'utf-8\'?> \n")
	fh.write("<q:quakeml xmlns:q=\"http://quakeml.org/xmlns/quakeml/1.2\" xmlns:ns0=\"http://service.iris.edu/fdsnws/event/1/\" xmlns=\"http://quakeml.org/xmlns/bed/1.2\"> \n")
	fh.write("  <eventParameters publicID=\"smi:local/6b269cbf-6b00-4643-8c2c-cbe6274083ae\"> \n")
	fh.write("    <event publicID=\"smi:service.iris.edu/fdsnws/event/1/query?eventid=3279407\"> \n")
	fh.write("      <preferredOriginID>smi:service.iris.edu/fdsnws/event/1/query?originid=9933375</preferredOriginID> \n")
	fh.write("      <preferredMagnitudeID>smi:service.iris.edu/fdsnws/event/1/query?magnitudeid=16642444</preferredMagnitudeID> \n")
	fh.write("      <type>earthquake</type> \n")
	fh.write("      <description ns0:FEcode=\"228\"> \n")
	fh.write("        <text>NEAR EAST COAST OF HONSHU, JAPAN</text> \n ")
	fh.write("        <type>Flinn-Engdahl region</type> \n")
	fh.write("      </description> \n")
	fh.write("      <origin publicID=\"smi:service.iris.edu/fdsnws/event/1/query?originid=9933375\" ns0:contributor=\"ISC\" ns0:contributorOriginId=\"02227159\" ns0:catalog=\"ISC\" ns0:contributorEventId=\"16461282\"> \n")
	fh.write("        <time> \n")
	fh.write("          <value>%s</value> \n" % tofe)
	fh.write("        </time> \n")
	fh.write("        <latitude> \n")
	fh.write("          <value>%f</value> \n" %lat)
	fh.write("        </latitude>\n")
	fh.write("        <longitude> \n")
	fh.write("          <value>%f</value> \n" %lon)
	fh.write("        </longitude>\n")
	fh.write("        <depth> \n")
	fh.write("          <value>%f</value> \n" %depth)
	fh.write("        </depth>\n")
	fh.write("        <creationInfo> \n")
	fh.write("          <author>Simon</author> \n")
	fh.write("        </creationInfo> \n")
	fh.write("      </origin> \n")
	fh.write("      <magnitude publicID=\"smi:service.iris.edu/fdsnws/event/1/query?magnitudeid=16642444\"> \n")
	fh.write("        <mag> \n")
	fh.write("          <value>9.1</value> \n")
	fh.write("        </mag> \n")
	fh.write("        <type>MW</type> \n")
	fh.write("        <originID>smi:service.iris.edu/fdsnws/event/1/query?originid=9933383</originID> \n")
	fh.write("        <creationInfo> \n")
	fh.write("          <author>Simon</author> \n")
	fh.write("        </creationInfo> \n")
	fh.write("      </magnitude> \n")
	fh.write("    </event> \n")
	fh.write("  </eventParameters> \n")
	fh.write("</q:quakeml>")



"""
Write station-files for synthetics
"""

with open( invfile, "w") as fh:
	fh.write("<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n")
	fh.write("<FDSNStationXML schemaVersion=\"1.0\" xmlns=\"http://www.fdsn.org/xml/station/1\">\n")
	fh.write("  <Source>IRIS-DMC</Source>\n")
	fh.write("  <Sender>IRIS-DMC</Sender>\n")
	fh.write("  <Created>2015-11-05T18:22:28+00:00</Created>\n")
	fh.write("  <Network code=\"LA\" endDate=\"2500-12-31T23:59:59+00:00\" restrictedStatus=\"open\" startDate=\"2003-01-01T00:00:00+00:00\">\n")
	fh.write("    <Description>Synthetic Array - Linear Array</Description>\n")
	fh.write("    <TotalNumberStations>20</TotalNumberStations>\n")
	fh.write("    <SelectedNumberStations>20</SelectedNumberStations>\n")

	if not uniform:
		station_range = randrange
	j=0
	for i in station_range:
		slon=i
		lat=54.0
		name="X"+str(int(j))
		fh.write("    <Station code=\"%s\" endDate=\"2011-11-17T23:59:59+00:00\" restrictedStatus=\"open\" startDate=\"2010-01-08T00:00:00+00:00\">\n" % name)
		fh.write("      <Latitude unit=\"DEGREES\">%f</Latitude>\n" % lat)
		fh.write("      <Longitude unit=\"DEGREES\">%f</Longitude>\n" % slon)
		fh.write("      <Elevation>0.0</Elevation>\n")
		fh.write("      <Site>\n")
		fh.write("        <Name> %s </Name>\n" % name)
		fh.write("      </Site>\n")
		fh.write("    <CreationDate>2010-01-08T00:00:00+00:00</CreationDate>\n")
		fh.write("    </Station>\n")
		j += 1
	fh.write("  </Network>\n")
	fh.write("</FDSNStationXML>")

inv=read_inv(invfile)
cat=read_cat(catfile)

st = stream.select(component="Z")
st.write( streamfile, format="Q")