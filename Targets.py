#  ======================== MONTE CARLO TREE SEARCH ========================== #
# Project:          Symbolic regression tests
# Name:             EvaluateFit.py
# Description:      Tentative implementation of a basic MCTS
# Authors:          Vincent Reverdy & Jean-Philippe Bruneton
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import numpy as np
import Build_dictionnaries
import Simplification_rules
import config
from scipy import interpolate

# ============================================================================ #

class Target:

    def __init__(self, which_target, maxsize, mode, fromfile = None):
        self.which_target = which_target
        self.mode = mode #'test' or 'train'
        self.from_file = fromfile
        self.maxsize = maxsize
        if self.from_file is None:
            self.target = self._define_target()
            self.mytarget = self.returntarget()
        else: #define target from file
            self.target = self._definetargetfromfile()
            self.mytarget = 'dummytarget'

    def _definetargetfromfile(self):
        n_targets = 1
        n_variables = 1
        maximal_size = self.maxsize
        if config.tryoscamorti:
            data = np.loadtxt('oscamorti.txt', delimiter=',')
        else:
            data = np.loadtxt(self.from_file, delimiter=',')
        L = data.shape[0]

        #si je coupe en deux:
        if False:
            lo = int(L*0.8)
            data_train = data[:lo, :]
            data_test = data[lo:, :]
            x_train = data_train[:, 0]
            x_test = data_test[:,0]
            f0_train = data_train[:,1]
            f0_test = data_test[:,1]

        else: # mieux je subsample genre 1 sur 2 quoi
            x_train = data[:, 0]
            f0_train = data[:, 1]
            x_test = data[:, 0]
            f0_test = data[:, 1]
            print('taille target', self.mode, len(x_train))

            tck = interpolate.splrep(x_test, f0_test, s=0)
            yder_test = interpolate.splev(x_test, tck, der=1)
            ysec_test = interpolate.splev(x_test, tck, der=2)

            tck = interpolate.splrep(x_train, f0_train, s=0)
            yder_train = interpolate.splev(x_train, tck, der=1)
            ysec_train = interpolate.splev(x_train, tck, der=2)

        print('check important', x_train.size, f0_train.size, yder_train.size, ysec_train.size)
        f_normalization_train = np.amax(np.abs(f0_train))
        #f_normalization_train = 1/(0.0006103515625)**2
        #f0_train = f0_train / f_normalization_train

        f_normalization_train = 1

        #f_normalization_test = np.amax(np.abs(f0_test))
        #f_normalization_test = 1/(0.0006103515625)**2
        #f0_test = f0_test / f_normalization_test
        f_normalization_test = 1
        range_x_train = x_train[-1] - x_train[0]
        range_x_test = x_test[-1] - x_test[0]
        #range_x_train = 1
        #range_x_test = 1
        if self.mode == 'train':
            return n_targets, n_variables, [x_train], [np.asarray(f0_train)], f_normalization_train, [range_x_train], maximal_size, [yder_train, ysec_train]
        else:
            return n_targets, n_variables, [x_test], [np.asarray(f0_test)], f_normalization_test, [range_x_test], maximal_size, [yder_test, ysec_test]


    def returntarget(self):
        with open('target_list.txt') as myfile:
            count = 0
            for line in myfile:
                if line[0] != '#' and line[0] != '\n':
                    if count == self.which_target:
                        mytarget = line
                        count+=1
                    else:
                        count+=1
        return mytarget

    def _define_target(self):
        ''' Initialize game : builds the target given by its number (of line) in target_list.txt '''
        # format of target_list.txt must be one target per line, and of the form:
        # n_targets, n_variables, expr1, train_set_type, train_set_range, test_set_type, test_set_range
        #todo : only works for one target now (not a system of eq)

        with open('target_list.txt') as myfile:
            count = 0
            for line in myfile:
                if line[0] != '#' and line[0] != '\n':
                    if count == self.which_target:
                        mytarget = line
                        count+=1
                    else:
                        count+=1
        print('Target function is: ', mytarget)
        mytarget = mytarget.replace(' ', '')
        mytarget = mytarget.replace('\n', '')
        mytarget = mytarget.split(',')
        n_targets = int(mytarget[0])
        n_variables = int(mytarget[1])
        target_function = mytarget[2] #is a string, ok
        maximal_size = int(mytarget[-1])

        # ----------------------------------#
        train_set_type_x = mytarget[3]

        # ----------------------------------#
        if train_set_type_x == 'E':
            train_set_range_x = [float(mytarget[4]), float(mytarget[5]), float(mytarget[6])]
            range_x_train = train_set_range_x[1] - train_set_range_x[0]

            x_train = np.linspace(train_set_range_x[0], train_set_range_x[1], num = int(range_x_train/train_set_range_x[2]))
            #following arxiv.1805.10365 where they give the spacing, not the number of points
        # or randomly sampled
        elif train_set_type_x == 'U':
            train_set_range_x = [float(mytarget[4]), float(mytarget[5]), int(mytarget[6])]
            range_x_train = train_set_range_x[1] - train_set_range_x[0]

            x_train = np.random.uniform(train_set_range_x[0], train_set_range_x[1], train_set_range_x[2])
            #re-order it!
            x_train = np.sort(x_train)
        else:
            print('training dataset not understood for target number', count -1)
            raise ValueError

        test_set_type_x = mytarget[7]

        if test_set_type_x == 'E':
            test_set_range_x = [float(mytarget[8]), float(mytarget[9]), float(mytarget[10])]
            range_x_test = test_set_range_x[1] - test_set_range_x[0]

            x_test = np.linspace(test_set_range_x[0], test_set_range_x[1], num=int((test_set_range_x[1]-test_set_range_x[0])/test_set_range_x[2]))
        # or randomly sampled
        elif test_set_type_x == 'U':
            test_set_range_x = [float(mytarget[8]), float(mytarget[9]), int(mytarget[10])]
            range_x_test = test_set_range_x[1] - test_set_range_x[0]

            x_test = np.random.uniform(test_set_range_x[0], test_set_range_x[1], test_set_range_x[2])
            x_test = np.sort(x_test)
        else:
            print('testing dataset not understood for target number', count - 1)
            raise ValueError

        # ----------------------------------#
        if n_variables > 1:
            train_set_type_y = mytarget[11]
            test_set_type_y = mytarget[15]

            if train_set_type_y == 'E':
                train_set_range_y = [float(mytarget[12]), float(mytarget[13]), float(mytarget[14])]
                range_y_train = train_set_range_y[1] - train_set_range_y[0]

                y_train = np.linspace(train_set_range_y[0], train_set_range_y[1], num=int(range_y_train/train_set_range_y[2]))
            # or randomly sampled
            elif train_set_type_y == 'U':
                train_set_range_y = [float(mytarget[12]), float(mytarget[13]), int(mytarget[14])]
                range_y_train = train_set_range_y[1] - train_set_range_y[0]

                y_train = np.random.uniform(train_set_range_y[0], train_set_range_y[1], train_set_range_y[2])
                y_train = np.sort(y_train)

            else:
                print('training dataset not understood for target number', count)
                raise ValueError

            if test_set_type_y == 'E':
                test_set_range_y = [float(mytarget[16]), float(mytarget[17]), float(mytarget[18])]
                range_y_test = test_set_range_y[1] - test_set_range_y[0]

                y_test = np.linspace(test_set_range_y[0], test_set_range_y[1], num=int((test_set_range_y[1]-test_set_range_y[0])/test_set_range_y[2]))
                # or randomly sampled
            elif test_set_type_y == 'U':
                test_set_range_y = [float(mytarget[16]), float(mytarget[17]), int(mytarget[18])]
                range_y_test = test_set_range_y[1] - test_set_range_y[0]

                y_test = np.random.uniform(test_set_range_y[0], test_set_range_y[1], test_set_range_y[2])
                y_test = np.sort(y_test)

            elif test_set_type_y == 'None':
                y_test = y_train
            else:
                print('testing dataset not understood for target number', count)
                raise ValueError

        # ----------------------------------#

        if n_variables > 2:
            train_set_type_z = mytarget[19]
            test_set_type_z = mytarget[23]


            if train_set_type_z == 'E':
                train_set_range_z = [float(mytarget[20]), float(mytarget[21]), float(mytarget[22])]
                range_z_train = train_set_range_z[1] - train_set_range_z[0]

                z_train = np.linspace(train_set_range_z[0], train_set_range_z[1], num=int(range_z_train/train_set_range_z[2]))
            # or randomly sampled
            elif train_set_type_z == 'U':
                train_set_range_z = [float(mytarget[20]), float(mytarget[21]), int(mytarget[22])]
                range_z_train = train_set_range_z[1] - train_set_range_z[0]
                z_train = np.random.uniform(train_set_range_z[0], train_set_range_z[1], train_set_range_z[2])
                z_train = np.sort(z_train)

            else:
                print('training dataset not understood for target number', count)
                raise ValueError

            if test_set_type_z == 'E':
                test_set_range_z = [float(mytarget[24]), float(mytarget[25]), float(mytarget[26])]
                range_z_test = test_set_range_z[1] - test_set_range_z[0]

                z_test = np.linspace(test_set_range_z[0], test_set_range_z[1], num=int((test_set_range_z[1]-test_set_range_z[0])/test_set_range_z[2]))
                # or randomly sampled
            elif test_set_type_z == 'U':
                test_set_range_z = [float(mytarget[24]), float(mytarget[25]), int(mytarget[26])]
                range_z_test = test_set_range_z[1] - test_set_range_z[0]

                z_test = np.random.uniform(test_set_range_z[0], test_set_range_z[1], test_set_range_z[2])
                z_test = np.sort(z_test)

            elif test_set_type_z == 'None':
                z_test = z_train
            else:
                print('testing dataset not understood for target number', count)
                raise ValueError

        # ------------------------------------------------#
        if n_variables == 1:
            x = x_train
            #print(x)
            #then i can eval f0
            f0_train = eval(target_function)
            f_normalization_train = np.amax(np.abs(f0_train))
            f_normalization_train = 1
            f0_train = f0_train/f_normalization_train

            x = x_test
            # then i can eval f0
            f0_test = eval(target_function)
            f_normalization_test = np.amax(np.abs(f0_test))
            f_normalization_test  = 1

            f0_test = f0_test/f_normalization_test

            #also schrink the x interval to -1, 1
            if self.mode == 'train':
                #return n_targets, n_variables, [x_train/range_x], [x_test/range_x], [f0_train], [f0_test], f_normalization, [range_x]
                return n_targets, n_variables, [x_train / range_x_train], [f0_train], f_normalization_train, [range_x_train], maximal_size
            else:
                return n_targets, n_variables,  [x_test/range_x_test], [f0_test], f_normalization_test, [range_x_test], maximal_size
        # ------------------------------------------------#
        elif n_variables == 2:

            # xtrain is a 1D array of x values
            # ytrain as well
            # but its easier to build variables X and Y as both arrays:

            X_train = np.zeros((x_train.size, y_train.size))
            Y_train = np.zeros((x_train.size, y_train.size))

            for i in range(x_train.size):
                X_train[i, :] = x_train[i]
            for j in range(y_train.size):
                Y_train[:, j] = y_train[j]

            X_test = np.zeros((x_test.size, y_test.size))
            Y_test = np.zeros((x_test.size, y_test.size))

            for i in range(x_test.size):
                X_test[i, :] = x_test[i]
            for j in range(y_test.size):
                Y_test[:, j] = y_test[j]

            x = X_train
            y = Y_train
            f0_train = eval(target_function)
            f_normalization_train = np.amax(np.abs(f0_train))
            f0_train = f0_train/f_normalization_train


            x = X_test
            y = Y_test
            f0_test = eval(target_function)
            f_normalization_test = np.amax(np.abs(f0_test))

            f0_test = f0_test/f_normalization_test

            if self.mode == 'train':
                #return n_targets, n_variables, [X_train/range_x, Y_train/range_y], [X_test/range_x, Y_test/range_y], [f0_train], [f0_test], f_normalization, [range_x, range_y]
                return n_targets, n_variables, [X_train/range_x_train, Y_train/range_y_train], f0_train, f_normalization_train, [range_x_train, range_y_train], maximal_size
            else:
                return n_targets, n_variables, [X_test/range_x_test, Y_test/range_y_test], f0_test, f_normalization_test, [range_x_test, range_y_test], maximal_size



        # ------------------------------------------------#
        elif n_variables == 3:

            X_train = np.zeros((x_train.size, y_train.size, z_train.size))
            Y_train = np.zeros((x_train.size, y_train.size, z_train.size))
            Z_train = np.zeros((x_train.size, y_train.size, z_train.size))

            for i in range(x_train.size):
                X_train[i, :, :] = x_train[i]
            for j in range(y_train.size):
                Y_train[:, j, :] = y_train[j]
            for k in range(z_train.size):
                Z_train[:, :, k] = z_train[k]

            X_test = np.zeros((x_test.size, y_test.size, z_test.size))
            Y_test = np.zeros((x_test.size, y_test.size, z_test.size))
            Z_test = np.zeros((x_test.size, y_test.size, z_test.size))

            for i in range(x_test.size):
                X_test[i, :, :] = x_test[i]
            for j in range(y_test.size):
                Y_test[:, j, :] = y_test[j]
            for k in range(z_test.size):
                Z_test[:, :, k] = z_test[k]

            x = X_train
            y = Y_train
            z = Z_train
            f0_train = eval(target_function)
            f_normalization_train = np.amax(np.abs(f0_train))
            f0_train = f0_train / f_normalization_train

            x = X_test
            y = Y_test
            z = Z_test
            f0_test = eval(target_function)
            f_normalization_test = np.amax(np.abs(f0_test))
            f0_test = f0_test / f_normalization_test
            if self.mode == 'train':
            #return n_targets, n_variables, [X_train/range_x, Y_train/range_y, Z_train/range_z], [X_test/range_x, Y_test/range_y, Z_test/range_z], [f0_train], [f0_test], f_normalization, [range_x, range_y, range_z]
                return n_targets, n_variables, [X_train / range_x_train, Y_train / range_y_train, Z_train / range_z_train], f0_train, f_normalization_train, [range_x_train, range_y_train, range_z_train], maximal_size
            else:
                return n_targets, n_variables, [X_test/range_x_test, Y_test/range_y_test, Z_test/range_z_test], f0_test, f_normalization_test, [range_x_test, range_y_test, range_z_test], maximal_size


class Voc():
    def __init__(self, target, modescalar):

        self.modescalar = modescalar
        self.target = target.target

        if self.modescalar == 'noA':
            self.maximal_size = 10+ self.target[-2]
        else:
            self.maximal_size = self.target[-2]

        self.numbers_to_formula_dict, self.arity0symbols, self.arity1symbols, self.arity2symbols, self.true_zero_number, self.neutral_element, \
        self.infinite_number, self.terminalsymbol, self.OUTPUTDIM, self.pure_numbers, self.arity2symbols_no_power, self.power_number, \
        self.arity0symbols_var_and_tar, self.var_numbers, self.plusnumber, self.minusnumber, self.multnumber, self.divnumber, self.log_number, \
        self.exp_number, self.explognumbers, self.trignumbers, self.sin_number, self.cos_number \
            = Build_dictionnaries.get_dic(self.target[0], self.target[1], modescalar)
        self.outputdim = len(self.numbers_to_formula_dict) - 3

        self.mysimplificationrules, self.maxrulesize = self.create_dic_of_simplifs()

    def replacemotor(self, toreplace,replaceby, k):
        firstlist = []
        secondlist = []
        for elem in toreplace:
            if elem == 'zero':
                firstlist.append(self.true_zero_number)
            elif elem == 'neutral':
                firstlist.append(self.neutral_element)
            elif elem == 'infinite':
                firstlist.append(self.infinite_number)
            elif elem == 'scalar':
                firstlist.append(self.pure_numbers[0])
            elif elem == 'mult':
                firstlist.append(self.multnumber)
            elif elem == 'plus':
                firstlist.append(self.plusnumber)
            elif elem == 'minus':
                firstlist.append(self.minusnumber)
            elif elem == 'div':
                firstlist.append(self.divnumber)
            elif elem == 'variable':
                firstlist.append(self.var_numbers[k])
            elif elem == 'arity0':
                firstlist.append(self.arity0symbols[k])
            elif elem == 'fonction':
                firstlist.append(self.arity1symbols[k])
            elif elem == 'allops':
                firstlist.append(self.arity2symbols[k])
            elif elem == 'power':
                firstlist.append(self.power_number)
            elif elem == 'log':
                firstlist.append(self.log_number)
            elif elem == 'exp':
                firstlist.append(self.exp_number)
            elif elem == 'sin':
                firstlist.append(self.sin_number)
            elif elem == 'cos':
                firstlist.append(self.cos_number)
            elif elem == 'one':
                firstlist.append(self.pure_numbers[0])
            elif elem == 'two':
                firstlist.append(self.pure_numbers[1])
            else:
                print('bug1', elem)

        for elem in replaceby:
            if elem == 'zero':
                secondlist.append(self.true_zero_number)
            elif elem == 'neutral':
                secondlist.append(self.neutral_element)
            elif elem == 'infinite':
                secondlist.append(self.infinite_number)
            elif elem == 'scalar':
                secondlist.append(self.pure_numbers[0])
            elif elem == 'mult':
                secondlist.append(self.multnumber)
            elif elem == 'plus':
                secondlist.append(self.plusnumber)
            elif elem == 'minus':
                secondlist.append(self.minusnumber)
            elif elem == 'div':
                secondlist.append(self.divnumber)
            elif elem == 'variable':
                secondlist.append(self.var_numbers[k])
            elif elem == 'arity0':
                secondlist.append(self.arity0symbols[k])
            elif elem == 'fonction':
                secondlist.append(self.arity1symbols[k])
            elif elem == 'allops':
                secondlist.append(self.arity2symbols[k])
            elif elem == 'empty':
                secondlist=[]
            elif elem == 'power':
                secondlist.append(self.power_number)
            elif elem == 'log':
                secondlist.append(self.log_number)
            elif elem == 'exp':
                secondlist.append(self.exp_number)
            elif elem == 'sin':
                secondlist.append(self.sin_number)
            elif elem == 'cos':
                secondlist.append(self.cos_number)
            elif elem == 'one':
                secondlist.append(self.pure_numbers[0])
            elif elem == 'two':
                secondlist.append(self.pure_numbers[1])
            else:
                print('bug2', elem)

        return firstlist, secondlist



    def create_dic_of_simplifs(self):

        if self.modescalar == 'A':
            mydic_simplifs = {}
            for x in Simplification_rules.mysimplificationrules_with_A:
                toreplace = x[0]
                replaceby = x[1]

                if 'variable' in toreplace:
                    for k in range(self.target[1]):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'arity0' in toreplace:
                    for k in range(len(self.arity0symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'fonction' in toreplace:
                    for k in range(len(self.arity1symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'allops' in toreplace:
                    for k in range(len(self.arity2symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))
                else:
                    firstlist, secondlist = self.replacemotor(toreplace, replaceby, 0)
                    mydic_simplifs.update(({str(firstlist): secondlist}))

            maxrulesize = 0
            for i in range(len(Simplification_rules.mysimplificationrules_with_A)):
                if len(Simplification_rules.mysimplificationrules_with_A[i][0]) > maxrulesize:
                    maxrulesize = len(Simplification_rules.mysimplificationrules_with_A[i][0])

            return mydic_simplifs, maxrulesize

        if self.modescalar == 'noA':
            mydic_simplifs = {}
            for x in Simplification_rules.mysimplificationrules_no_A:
                toreplace = x[0]
                replaceby = x[1]
                if 'variable' in toreplace:
                    for k in range(self.target[1]):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'arity0' in toreplace:
                    for k in range(len(self.arity0symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'fonction' in toreplace:
                    for k in range(len(self.arity1symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'allops' in toreplace:
                    for k in range(len(self.arity2symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                else:
                    firstlist, secondlist = self.replacemotor(toreplace, replaceby, 0)
                    mydic_simplifs.update(({str(firstlist): secondlist}))

            maxrulesize = 0

            for i in range(len(Simplification_rules.mysimplificationrules_no_A)):
                if len(Simplification_rules.mysimplificationrules_no_A[i][0]) > maxrulesize:
                    maxrulesize = len(Simplification_rules.mysimplificationrules_no_A[i][0])

            return mydic_simplifs, maxrulesize