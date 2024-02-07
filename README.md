# Seismic-Toolbox: bowPy
Development of a toolbox applicable to seismological array data using Body Waves, containing f-k methods for data reconstruction and filtering


Installation:

- clone/download the repo

- cd into bowPy

- python3 -m venv env

- source env/bin/activate

- (env) pip install .

- (env) pip install basemap

start ipython

voilá


Example of FK-filter
```
from obspy.core.stream import read as read_stream
from obspy.core.inventory import read_inventory
from obspy.core.event import read_events
from bowpy.filter.fk import fk_filter
from bowpy.util.array_util import vespagram


st = read_stream("PATH_TO_DATA/*.SAC")
inv = read_inventory("PATH_TO_INVENTORY_FILE")
events = read_events("PATH_TO_EVENT_FILE")

vespa_data = vespagram(st, inv=inv, event=events[0])

fitered_st = fk_filter(st, inv=inv, event=events[0])

vespa_data_filtered = vespagram(fitered_st, inv=inv, event=events[0])

```