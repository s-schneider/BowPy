from bowpy.util.data_request import data_request
from obspy.clients.fdsn import Client
import obspy

client = Client('IRIS')

events = client.get_events(minmagnitude=8)

cat = obspy.core.event.Catalog()
cat.append(events[29])
data_request('IRIS', cat=cat, channels="VHE,VHN,VHZ",
             savefile='station', normal_mode_data=True,
             file_format='ah')
