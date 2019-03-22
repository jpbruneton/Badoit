#!/bin/bash

# ./runDockerExpe.sh config priority executionMode image hostnames
# Parameters:
#  first parameter: name of the configuration file (default "conf/configuration.yaml")
#  second parameter: priority (low, mid, high, veryhigh). Please DO NOT use "veryhigh" without authorisation.
#  third parameter: execution mode (normal or cluster).
#  fourth parameter: name of the docker image (default "shoal-python-pi:latest")
#  fifth parameter: list of hosts to launch the images, if execution mode is set to 'cluster'. The list must be split by ",". E.g.: "shoal1,shoal2,shoal3,shoal4,shoal5,shoal6,shoal7,shoal8"

configFile=${1:-conf/configuration.yaml}
priorityLevel=${2:-"low"}
executionMode=${3:-"normal"}
imageName=${4:-"shoal-python-pi:latest"}
hostsNames=${5:-"shoal1,shoal2,shoal3,shoal4,shoal5,shoal6,shoal7,shoal8"}

expeName="shoal-python-pi" # TO ADAPT
networkName="net-$expeName"
memoryLimit=8G
resultsPath=$(pwd)/results
resultsPathInContainer=/home/user/results
uid=$(id -u)
confPath=$(pwd)/conf
confPathInContainer=/home/user/conf

if [ ! -d $resultsPath ]; then
    mkdir -p $resultsPath
fi

inDockerGroup=`id -Gn | grep docker`
if [ -z "$inDockerGroup" ]; then
    sudoCMD="sudo"
else
    sudoCMD=""
fi
dockerCMD="$sudoCMD docker"

if [ -d "$confPath" ]; then
    confVolParam="-v $confPath:$confPathInContainer"
else
    confVolParam=""
fi

if [ "$priorityLevel" = "low" ]; then
	priorityParam="-c 128"
elif [ "$priorityLevel" = "mid" ]; then
	priorityParam="-c 512"
elif [ "$priorityLevel" = "high" ]; then
	priorityParam="-c 2048"
elif [ "$priorityLevel" = "veryhigh" ]; then
	priorityParam="-c 131072"
else
	priorityParam="-c 128"
fi


if [ "$executionMode" = "normal" ]; then
    exec $dockerCMD run -i -m $memoryLimit --rm $priorityParam -v $resultsPath:$resultsPathInContainer $confVolParam $imageName  "$uid" "normal" "-c $configFile"
elif [ "$executionMode" = "cluster" ]; then
    # Create host file
    hostsFilename=$resultsPath/hosts
    declare -a hosts
    IFS="," read -r -a hosts <<< "$hostsNames"
    echo -n > $hostsFilename
    for h in "${hosts[@]}"; do
        echo "$expeName-$h" >> $hostsFilename
    done

    # Create network
    $dockerCMD network create --driver=overlay --attachable $networkName
    sleep 1

    # Launch a client on each node
    for h in "${hosts[@]}"; do
        name="$expeName-$h"
        rm $resultsPath/stop-$h
        ssh $h "$dockerCMD run -di --name $name --network $networkName -m $memoryLimit --rm $priorityParam -v $resultsPath:$resultsPathInContainer $confVolParam  $imageName $uid client $resultsPathInContainer/stop-$h"
    done
    sleep 1

    # Launch a server node
    $dockerCMD run -i --name $expeName-main --network $networkName -m $memoryLimit --rm $priorityParam -v $resultsPath:$resultsPathInContainer $confVolParam  $imageName "$uid" "server" $resultsPathInContainer/hosts "-c $configFile -p scoop"

    # Quit all client nodes
    for h in "${hosts[@]}"; do
        echo -n > $resultsPath/stop-$h
    done

    # Remove network
    $dockerCMD network rm $networkName
fi


# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
