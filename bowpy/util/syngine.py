from obspy.clients.syngine import Client

# Example
client = Client()

stream = client.get_waveforms(model='ak135f_1s', network='XD', station='A*', eventid=)
