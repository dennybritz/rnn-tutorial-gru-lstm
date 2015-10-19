#! /usr/bin/env bash

mkdir -p data/glove
curl http://www-nlp.stanford.edu/data/glove.6B.100d.txt.gz | gunzip > data/glove/glove.6B.100d.txt