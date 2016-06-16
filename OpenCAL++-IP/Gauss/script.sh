#!/bin/bash

for entry in "$search_dir"./input/tiff/*
do

    arrIN=(${entry//\// })
   ./bin/gauss ${arrIN[3]}
  
done
