#!/bin/sh
path="`python -m site --user-site`"

mkdir -p $path

cp -r dev/sipy $path/.


