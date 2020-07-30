#!/bin/bash
set -e

uid=$1 #${1:-1000}
executionMode=$2 #${2:-"normal"}
shift; shift;

baseDir="/home/user/Badoit"
baseCmd="python3 run_all_targets.py $@"

useradd -d /home/user -Ms /bin/bash -u $uid user
chown -R $uid /home/user

# Launch openssh server
/etc/init.d/ssh start

# Launch program
exec gosu user bash -c "cd $baseDir; $baseCmd"

if [ "$executionMode" = "normal" ]; then
    exec gosu user bash -c "cd $baseDir; $baseCmd"
elif [ "$executionMode" = "client" ]; then
    stopFilePath=$1 #${3:-"/home/user/results/stop-$(hostname)"}
    shift;
    while [ ! -f $stopFilePath ]; do
        sleep 1
    done
    rm $stopFilePath
    exit
elif [ "$executionMode" = "server" ]; then
    hostsFilePath=$1 #${3:-"/home/user/results/hosts"}
    shift;
    hosts=`cat $hostsFilePath | cut -f 1 -d ' '`
    gosu user bash -c "touch /home/user/.ssh/known_hosts"
    for h in $hosts; do
        gosu user bash -c "ssh-keyscan $h >> /home/user/.ssh/known_hosts"
    done
    exec gosu user bash -c "cd $baseDir; python3 -m scoop --hostfile $hostsFilePath $baseCmd -p scoop"
fi


# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
