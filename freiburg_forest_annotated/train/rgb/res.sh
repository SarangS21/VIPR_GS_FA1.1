#!/bin/bash

for i in *.png; do
    printf "resize $i\n"
    convert "$i" -resize 720 960 "$i"
done
