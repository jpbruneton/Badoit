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
from game_env import Game
from State import State
import numpy as np
from MCTS import MCTS
import config
import time
import pickle
from multiprocessing import Process
import tqdm
import os
import ResNet1D
import torch
from ResNet1D import ResNet_Training
from game_env import game_evaluate



# ======================= Useful fonctions for MAIN ========================== #

def load_or_create_neural_net(maxL, outputdim):
    if config.net == 'resnet':
        file_path = './best_model_resnet.pth'
    elif config.net == 'densenet':
        file_path = './best_model_densenet.pth'
    else:
        print('Neural Net type not understood. Chose between resnet or densenet in config.py')
        raise ValueError

    if config.net == 'resnet':
        if os.path.exists(file_path):
            print('loading already trained model')
            time.sleep(0.3)

            best_player_so_far = ResNet1D.resnet18(maxL, outputdim)
            best_player_so_far.load_state_dict(torch.load(file_path))
            best_player_so_far.eval()

        else:
            print('Trained model doesnt exist. Starting from scratch.')
            time.sleep(0.3)

            best_player_so_far = ResNet1D.resnet18(maxL, outputdim)
            best_player_so_far.eval()

    if config.net == 'densenet':
        if os.path.exists(file_path):
            print('loading already trained model')
            time.sleep(0.3)

            best_player_so_far = ResNet1D.densenet()
            best_player_so_far.load_state_dict(torch.load(file_path))
            best_player_so_far.eval()

        else:
            print('Trained model doesnt exist. Starting from scratch.')
            time.sleep(0.3)
            best_player_so_far = ResNet1D.densenet()
            best_player_so_far.eval()

    return best_player_so_far

def evalQ(best_player_so_far, generate_datas,maxL):
    dataset_bad = generate_datas.sample(500, with_mean=-0.5)[:, 0, 0:maxL]
    dataset_neutral = generate_datas.sample(500, with_mean=0)[:, 0, 0:maxL]
    dataset_good = generate_datas.sample(500, with_mean=0.5)[:, 0, 0:maxL]

    meanqval_bad = 0
    meanqval_neutral = 0
    meanqval_good = 0

    best_player_so_far.cpu()
    for i in range(500):
        Qvalue_bad, _ = best_player_so_far.forward(dataset_bad[i])
        Qvalue_neutral, _ = best_player_so_far.forward(dataset_neutral[i])
        Qvalue_good, _ = best_player_so_far.forward(dataset_good[i])

        meanqval_bad += Qvalue_bad.detach().numpy()[0] / 500
        meanqval_neutral += Qvalue_neutral.detach().numpy()[0] / 500
        meanqval_good += Qvalue_good.detach().numpy()[0] / 500

    print('resultats Q accuracy', meanqval_bad, meanqval_neutral, meanqval_good)



def improve_model(player, previous_data, pretrain=False):
    np.random.seed(int(100000*time.time())%100)
    min_data = config.MINIBATCH * config.MINBATCHNUMBER
    max_data = config.MINIBATCH * config.MAXBATCHNUMBER
    size_training_set = previous_data.shape[0]

    if pretrain==True:
        tailleset = previous_data.shape[0]
        X = previous_data[-tailleset:-1, :]
        training = ResNet_Training(player, config.MINIBATCH, config.EPOCHS, config.LEARNINGRATE, X, X, 1)
        training.trainNet()
    else:
        if size_training_set >= max_data:
            X = previous_data[np.random.choice(previous_data.shape[0], max_data, replace=False)]
            #X = data[-size_training_set:-1,:]
        else:
            X=previous_data

            #if tailleset>maxsize:
            #    tailleset=maxsize

            #preX = previous_data[-tailleset:-1,:]
            #idx=np.random.randint(maxsize-1, size=int(maxsize/15))
            #X=preX[idx, :]

        training = ResNet_Training(player,config.MINIBATCH,config.EPOCHS,config.LEARNINGRATE,X,X,1)
        training.trainNet()
        # else:
        #     print('playing more games to have more training data')
        #     print('train set', previous_data.shape[0])



def generate_self_play_data(aggro, maxL, outputdim, voc, tau,tauzero,target,rewardmin, tolerance, best_player_so_far, allformulaseen_self_play):

    player_reward_self_play = 0
    avQ = 0
    previous_data = np.zeros(maxL + outputdim + 1).reshape((1, 1, maxL + outputdim + 1))
    rollouts=[]
    combo = True
    if aggro:
        iter = config.loop_agg_play
    else:
        iter = config.loop_self_play
    for _ in tqdm.tqdm(range(iter)):

        procs = []

        for index in range(config.cpus):
            proc = Process(target=play_parrallel_pytorch,
                           args=(aggro, maxL, outputdim, voc, config.CPUCT, config.SIM_NUMBER, target, rewardmin, tolerance, best_player_so_far, tau, tauzero, combo, index,))


            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        # retrieve results of parrallel subprocesses
        checkallfor=[]
        for index in range(config.cpus):
            filename = 'results' + str(index) + '.txt'
            with open(filename, 'rb') as file:
                load_dic = pickle.load(file)

            newrew, NN_thought, new_data, newformula, newstate, rolloutsformula = load_dic['result']
            player_reward_self_play += newrew / (config.cpus*iter)
            previous_data = np.vstack((previous_data, new_data))
            checkallfor.append(newformula)
            #if config.add_rd_eqs :
#                add_rd_data = generate_datas.sample(int(config.SENTENCELENGHT / 12), 0)
#                previous_data = np.vstack((previous_data, add_rd_data))

            avQ += NN_thought / (config.cpus*iter)
            allformulaseen_self_play.append([newrew, newformula, newstate])
            rollouts.extend(rolloutsformula)
            file.close()
    time.sleep(0.1)
    print('done self play (tau=', tau, '), with av reward of',
          player_reward_self_play, 'versus NN perceived value', avQ)
    print('for the equation', newformula)
    print('all aggro form', checkallfor)


    dataseen = np.delete(previous_data, 0 , 0)

    return  allformulaseen_self_play, rollouts, player_reward_self_play, dataseen


def generate_agg_play_data(maxL, outputdim, tau,tauzero,target,rewardmin, tolerance, best_player_so_far, allformulaseen_self_play):

    player_reward_self_play = 0
    avQ = 0
    previous_data = np.zeros(maxL + outputdim + 1).reshape((1, 1, maxL + outputdim + 1))
    rollouts=[]
    combo = False
    for _ in tqdm.tqdm(range(config.loop_agg_play)):

        procs = []

        for index in range(config.cpus):
            proc = Process(target=play_parrallel_pytorch,
                           args=(config.CPUCT, config.SIM_NUMBER, target, rewardmin, tolerance, best_player_so_far, tau, tauzero, combo, index,))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        # retrieve results of parrallel subprocesses

        for index in range(config.cpus):
            filename = 'results' + str(index) + '.txt'
            with open(filename, 'rb') as file:
                load_dic = pickle.load(file)

            newrew, NN_thought, new_data, newformula, newstate, rolloutsformula = load_dic['result']
            player_reward_self_play += newrew / (config.cpus*config.loop_self_play)
            previous_data = np.vstack((previous_data, new_data))

            if config.add_rd_eqs :
                add_rd_data = generate_datas.sample(int(config.SENTENCELENGHT / 12), 0)
                previous_data = np.vstack((previous_data, add_rd_data))

            avQ += NN_thought / (config.cpus*config.loop_self_play)
            allformulaseen_self_play.append([newrew, newformula, newstate])
            rollouts.extend(rolloutsformula)
            file.close()
    time.sleep(0.1)
    print('done self play (tau=', tau, '), with av reward of',
          player_reward_self_play, 'versus NN perceived value', avQ)
    print('for the equation', newformula)


    dataseen = np.delete(previous_data, 0 , 0)

    return  allformulaseen_self_play, rollouts, player_reward_self_play, dataseen


def play_parrallel_pytorch(aggro, maxL, outputdim, voc, cpuct, sim_number, target, rewardmin, tolerance, player, tau, tauzero, combo, index,):

    # init tree
    game = Game(voc)
    initialstate = game.state
    mcts_tree = MCTS(maxL, outputdim, voc, player, target, rewardmin, tolerance)
    currentnode = mcts_tree.createNode(initialstate)

    #check evol
    nninput = currentnode.state.convert_to_NN_input()

    NNreward, NNproba_children = player.forward(nninput)
    # print('pc0', proba_children)
    checkevol = True
    reward = NNreward.detach().numpy()[0][0]
    proba_children = NNproba_children.detach().numpy()[0]
    if index==1 and checkevol:
        print('begning game')
        print(reward)
        print(proba_children)
    # counters
    turn = 0
    isterminal = 0

    # misc
    lenstate=maxL
    lenpi=outputdim
    dataseen = np.zeros(lenstate + lenpi + 1)

    #----------------------------------------------------------#
    #Main loop
    found_eq = 0
    ext_eq = 0

    while isterminal == 0:

        turn = turn + 1

        # check evol
        if turn ==2 and index == 1 and checkevol:
            nninput = currentnode.state.convert_to_NN_input()

            NNreward, NNproba_children = player.forward(nninput)
            # print('pc0', proba_children)

            reward = NNreward.detach().numpy()[0][0]
            proba_children = NNproba_children.detach().numpy()[0]
            print('turn2 ')
            print(reward)
            print(proba_children)

        # check evol
        if turn == 3 and index == 1 and checkevol:
            nninput = currentnode.state.convert_to_NN_input()

            NNreward, NNproba_children = player.forward(nninput)
            # print('pc0', proba_children)

            reward = NNreward.detach().numpy()[0][0]
            proba_children = NNproba_children.detach().numpy()[0]
            print('turn3 ')
            print(reward)
            print(proba_children)


        np.random.seed(index + int(10000000 * time.time() % 1000))
        for sims in range(0, sim_number):
            mcts_tree.simulate(currentnode, cpuct, combo)

        visits_after_all_simulations=np.array([])
        childactions = np.array([])
        for child in currentnode.children:
            visits_after_all_simulations=np.hstack((visits_after_all_simulations, child.N**(1/tau)))
            childactions=np.hstack((childactions, child.char -1))

        probvisit=visits_after_all_simulations/np.sum(visits_after_all_simulations)

        childactions=np.asarray(childactions, dtype=int)

        unmask_pi = np.zeros(lenpi)
        unmask_pi[childactions] = probvisit

        createdata = np.zeros(lenstate + lenpi + 1).reshape(1, lenstate + lenpi + 1)

        # we need to renormalize the input!!
        nninput = currentnode.state.convert_to_NN_input()
        createdata[0, 0:lenstate] = nninput
        createdata[0, lenstate:lenstate + lenpi] = unmask_pi

        dataseen = np.vstack((dataseen, createdata))

        #MOVE DOWN : take a step

        if turn < tauzero:
            currentnode = np.random.choice(currentnode.children, p=probvisit)
        else:
            max = np.random.choice(np.where(visits_after_all_simulations == np.max(visits_after_all_simulations))[0])
            currentnode = currentnode.children[max]

        isterminal = currentnode.isterminal()

        if isterminal == 1:
            #print(currentnode.state.formulas)
            game = Game(voc, currentnode.state)
            reward, _, _ = game_evaluate(game.state.reversepolish, game.state.formulas, tolerance, voc, target, "train")

            nninput = currentnode.state.convert_to_NN_input()

            NN_thought, _ = player.forward(nninput)

            NN_thought = NN_thought.detach().numpy()[0][0]

            newdata = dataseen
            newdata[:,-1] = reward

            newdata = np.delete(newdata, 0, 0)

            newdata = newdata.reshape(newdata.shape[0],1, newdata.shape[1])

        else:
            #if not reinit tree
            game = Game(voc, currentnode.state)
            initialstate = game.state
            mcts_tree = MCTS(maxL, outputdim, voc, player, target, rewardmin, tolerance)
            currentnode = mcts_tree.createNode(initialstate)

    save_dic= {}
    save_dic['result']=[reward,  NN_thought, newdata, game.state.formulas, game.state.reversepolish, mcts_tree.saveterminals]
    filename = 'results' + str(index) + '.txt'
    with open(filename, 'wb') as file:
        pickle.dump(save_dic, file)

    file.close()



def drawparams(params):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.arange(params[0].size)
    y0 = params[0]
    y1 = params[1]
    y2 = params[2]

    print(x.size)
    ax.plot(x,y0,'g-',label='weight 1')
    ax.plot(x,y1,'r-',label='weight 2')
    ax.plot(x,y2,'b-',label='weight 3')

    ax.set_title('evolution des 3 poids pris random du NN')
    ax.set_xlabel('nombre d itérations')
    ax.set_ylabel('valeur des poids')
    ax.legend()

    plt.savefig("evolution_poids")

    plt.show()

