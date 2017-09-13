from bowpy.util.data_request import data_request
from obspy.clients.fdsn import Client

client = Client('IRIS')

cat = client.get_events(minmagnitude=8)
st, inv, cat = data_request('IRIS', cat=cat, net='*', channels='*',
                            savefile=True, normal_mode_data=True,
                            file_format='ah')
