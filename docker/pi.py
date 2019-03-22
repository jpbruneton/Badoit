#!/usr/bin/env python3


"""
Calculation of Pi using a Monte Carlo method. Using code from the "pi_calc.py" official example of the scoop library (https://github.com/soravux/scoop/blob/master/examples/pi_calc.py) under the GPLv3 licence.
"""

from math import hypot
from random import random
from time import time


def init_parallelism(parallelismType):
    if parallelismType == "none":
        pass
        mapFn = map
    elif parallelismType == "multiprocessing":
        import multiprocessing
        global pool
        pool = multiprocessing.Pool()
        mapFn = pool.map
    elif parallelismType == "scoop":
        import scoop
        mapFn = scoop.futures.map
    else:
        raise ValueError("Unknown parallelismType: '%s'" % parallelismType)
    return mapFn

def stop_parallelism(parallelismType):
    if args.parallelismType == "none":
        pass
    elif args.parallelismType == "multiprocessing":
        global pool
        pool.close()
    elif args.parallelismType == "scoop":
        pass


def test(tries):
    return sum(hypot(random(), random()) < 1 for _ in range(tries))

# Calculates pi with a Monte-Carlo method. This function calls the function
# test "n" times with an argument of "t".
def calcPi(workers, tries, mapFn):
    bt = time()
    expr = mapFn(test, [tries] * workers)
    piValue = 4. * sum(expr) / float(workers * tries)
    totalTime = time() - bt
    print("pi = " + str(piValue))
    print("total time: " + str(totalTime))
    return piValue


########## MAIN ########### {{{1
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configFilename', type=str, default='conf/configuration.yaml', help = "Path of configuration file")
    parser.add_argument('-o', '--outputDir', type=str, default='results/', help = "Path of output files")
    parser.add_argument('-p', '--parallelismType', type=str, default='multiprocessing', help = "Type of parallelism to use")
    args = parser.parse_args()

    # Load config
    import yaml
    configFilename = args.configFilename
    config = yaml.safe_load(open(configFilename))

    # Init parallelism
    parallelismType = args.parallelismType
    mapFn = init_parallelism(parallelismType)

    # Estimate pi
    dataPi = calcPi(config['workers'], config['tries'], mapFn)

    # Stop parallelism
    stop_parallelism(parallelismType)


# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
