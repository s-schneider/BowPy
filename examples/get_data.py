from bowpy.util.data_request import data_request
from obspy.clients.fdsn import Client
import obspy
import os

path = '/data/simons/3-comp-data'


client = Client('IRIS')

events = client.get_events(minmagnitude=8)
oldfolder = None
for event in events[::-1]:
    eventy = str(event.origins[0].time.year)
    eventm = str(event.origins[0].time.month)
    eventd = str(event.origins[0].time.day)
    eventh = str(event.origins[0].time.hour)
    newfolder = '/' + eventy + '-' + eventm + '-' + eventd
    if newfolder == oldfolder:
        newfolder = newfolder + 'B'
    oldfolder = newfolder

    newpath = path + newfolder
    print(newpath)
    os.mkdir(newpath)
    os.chdir(newpath)
    cat = obspy.core.event.Catalog()
    cat.append(event)
    data_request('IRIS', cat=cat, channels="VHE,VHN,VHZ,LHE,LHN,LHZ",
                 savefile='station', normal_mode_data=True,
                 file_format='ah')
