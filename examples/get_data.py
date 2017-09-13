from bowpy.util.data_request import data_request
from obspy.clients.fdsn import Client

client = Client('IRIS')

cat = client.get_events(minmagnitude=8)
station = '034A'
st, inv, cat = data_request('IRIS', cat=cat, scode=station, net='*', channels='*', savefile=False,
             normal_mode_data=True, file_format='ah')
