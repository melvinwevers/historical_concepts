#!/bin/sh

python3 making_embeddings_newspapers_new.py --y0=1950 --yN=1994 --nYears=5 --title=ah_nrc --outdir=../embeddings_new --step=1
python3 making_embeddings_newspapers_new.py --y0=1950 --yN=1994 --nYears=10 --title=ah_nrc --outdir=../embeddings_new --step=5
python3 making_embeddings_newspapers_new.py --y0=1950 --yN=1994 --nYears=40 --title=ah_nrc --outdir=../embeddings_new --step=10
