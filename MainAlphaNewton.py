#  =========== MONTE CARLO TREE SEARCH GUIDED BY NEURAL NETWORK ============== #
# Project:          Symbolic regression tests
# Name:             Game.py
# Description:      Tentative implementation of a basic MCTS
# Authors:          Adèle Douin & Vincent Reverdy & Jean-Philippe Bruneton
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages

import time
import numpy as np
import config
import pickle
from operator import itemgetter
from Targets import Target, Voc
import main_functions
from multiprocessing import Process
from game_env import Game
import torch
import torch.utils
import torch.utils
from torchsummary import summary
import copy
import exec_gp_qd
#import Geneticc_post_treatment


# =================================== MAIN ==================================== #


def launch():

    # --------------------------------------------------------------------- #
    # Init target
    which_target = 0
    target = Target(which_target, 'train') #this is train target
    test_target = Target(which_target, 'test')
    rewardmin =-1


    # init dictionnaries
    voc_no_a = Voc(target, 'noA')
    print('working with: ', voc_no_a.numbers_to_formula_dict)
    maxL = voc_no_a.maximal_size
    outputdim = voc_no_a.outputdim


    # --------------------------------------------------------------------- #
    #init tolerance
    tolerance = exec_gp_qd.init_tolerance(target, voc_no_a)


    # init data --------------------------------------------------------------------- #
    alleqs = []

    allformulaseen_self_play = []
    allformulaseen_aggro = []
    allrollouts=[]


    # --------------------------------------------------------------------- #
    # Init NN
    best_player_so_far = main_functions.load_or_create_neural_net(maxL, outputdim)

    # --------------------------------------------------------------------- #
    # print summary
    printsummary = 0
    if printsummary:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if config.net == 'densenet':
            print(summary(best_player_so_far.to(device), (1, voc_no_a.maximal_size)))
        if config.net == 'resnet':
            print(summary(best_player_so_far.to(device), (1, voc_no_a.maximal_size)))
        time.sleep(0.1)

     # --------------------------------------------------------------------- #
    # MAIN LOOP FOR SINGLE PLAYER
    i=0

    maxiter = 100
    allavreward=[-1]
    allavrewardagg = [-1]

    while i < maxiter:

        #revoir improve condition
        #previous_best = copy.deepcopy(best_player_so_far)

        print('')
        print('this is iteration number', i)


        print('generate data with self_play')
        time.sleep(0.1)
        #mp.set_start_method('spawn')
        tau = 1
        tauzero=int(2*maxL/4)

        allformulaseen_self_play, rollouts, player_reward_self_play, dataseen = \
            main_functions.generate_self_play_data(False, maxL, outputdim, voc_no_a, tau, tauzero, target,
                                                   rewardmin, tolerance, best_player_so_far, allformulaseen_self_play)


        print('')
        print('at the end of generating data')
        print('we have seen', len(allformulaseen_self_play), 'different eqs')
        bestnn=sorted(allformulaseen_self_play, key=itemgetter(0), reverse=True)[0:2]
        print('and best one is', [bestnn[0][0],bestnn[0][1],bestnn[1][0],bestnn[1][1]])

        time.sleep(0.2)
        #print('we have seen', len(rollouts), 'different rollouts eqs')
        #bestnn = sorted(rollouts, key=itemgetter(0), reverse=True)[0:2]
        #print('and best one is', [bestnn[0][0], bestnn[0][1], bestnn[1][0], bestnn[1][1]])

        print('')
        print('---now improving model----')

        main_functions.improve_model(best_player_so_far, dataseen)

        print('done training')
        time.sleep(2)

        print('')
        print('..aggro play')

        #check aggresive play:
        tau = 0.1
        tauzero = 1
        allformulaseen_aggro, rollouts, player_reward_self_play, dataseen = \
            main_functions.generate_self_play_data(True, maxL, outputdim, voc_no_a, tau, tauzero, target,
                                                   rewardmin, tolerance, best_player_so_far, allformulaseen_aggro)

        print('')
        print('----------------------')
        print('at the end of aggressive play')
        print('we have seen', len(allformulaseen_aggro), 'different eqs')
        bestnn = sorted(allformulaseen_aggro, key=itemgetter(0), reverse=True)[0:2]
        print('and best one is', [bestnn[0][0], bestnn[0][1], bestnn[1][0], bestnn[1][1]])

        #print('we have seen', len(rollouts), 'different rollouts eqs')
        #bestnn = sorted(rollouts, key=itemgetter(0), reverse=True)[0:2]
        #print('and best one is', [bestnn[0][0], bestnn[0][1], bestnn[1][0], bestnn[1][1]])

        if config.net == 'resnet':
            torch.save(best_player_so_far.state_dict(), './best_model_resnet.pth')

        #---------------------------------------------------------------------#
        #end main loop
        i+=1



if __name__ == '__main__':
    launch()

# revoir le improve condition :
# allavreward.append(player_reward_self_play)
#         print('results av reward', allavreward)
#         allrollouts.extend(rollouts)
#         time.sleep(0.5)
#
#         if i>0:
#             if allavreward[-1] > allavreward[-2]:
#                 if config.net == 'densenet':
#                     torch.save(best_player_so_far.state_dict(), './best_model_densenet.pth')
#                 if config.net == 'resnet':
#                     torch.save(best_player_so_far.state_dict(), './best_model_resnet.pth')
#             else:
#                 best_player_so_far = previous_best
#                 print('model has not improved enough')
#                 best_player_so_far.eval()
#
# # --------------------------------------------------------------------- #
# # Plot the results
# if mode == 'oneplayer':
#     plt.plot(av_reward_of_deterministic_play, color='red')
#     plt.plot(av_reward_of_newdata, color='black')
#     plt.savefig("evolution_av_reward")
#
#     plt.show()
#     drawparams(params)
#
#
# if mode == 'twoplayers':
#     plt.plot(av_reward_of_deterministic_play_player1, color='blue')
#     plt.plot(av_reward_of_deterministic_play_player2, color='green')
#
#     plt.plot(av_reward_of_newdata, color='black')
#
#     plt.show()