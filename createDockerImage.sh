#!/bin/sh

imageName=${1:-jpbruneton/badoit:latest}

inDockerGroup=`id -Gn | grep docker`
if [ -z "$inDockerGroup" ]; then
	sudoCMD="sudo"
else
	sudoCMD=""
fi
dockerCMD="$sudoCMD docker"
dockerParams="--no-cache"

$dockerCMD build $dockerParams -t $imageName .

