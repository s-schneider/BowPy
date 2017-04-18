#!/bin/sh
path="`python -m site --user-site`"

mkdir -p $path

cp -r sipy $path/.

cp startup.py ~/.ipython/profile_default/startup/.


