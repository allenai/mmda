#!/bin/bash

set -e

DIR=$(cd `dirname $0` && pwd)

docker-compose -f $DIR/docker/pipeline.yml build
docker-compose -f $DIR/docker/pipeline.yml up --detach symbolscraper layoutparser vila
docker-compose -f $DIR/docker/pipeline.yml run -v $DIR:/pipeline python /pipeline/run.py /pipeline/test.pdf
docker-compose -f $DIR/docker/pipeline.yml down
