#!/bin/bash

FILE=${SYNTH-""}

depth=100.0
lat=0.0
lon=0.0
origin="09-FEB-2016_18:41:01.0"

shc<<EOF
	read $FILE all
	set all depth $depth
	set all lat $lat
	set all lon $lon
	set all origin $origin
	epi_dist $lat $lon
	quit y
EOF
