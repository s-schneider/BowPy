#!/bin/sh
path="`python -m site --user-site`"

mkdir -p $path

cp -r bowpy $path/.

cp startup.py ~/.ipython/profile_default/startup/.
