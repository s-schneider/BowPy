from bowpy.util.data_request import data_request
from obspy.clients.fdsn import Client

client = Client('IRIS')

events = client.get_events(minmagnitude=8)

cat = obspy.core.event.Catalog()
cat.append(events[0])
st, inv, cat = data_request('IRIS', cat=cat, net='*', channels='*',
                            savefile='network', normal_mode_data=True,
                            file_format='ah')
