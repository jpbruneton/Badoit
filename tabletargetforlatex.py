def returntarget(u):
    with open('target_list.txt') as myfile:
        count = 0
        for line in myfile:
            if line[0] != '#' and line[0] != '\n':
                if count == u:
                    mytarget = line
                    count += 1
                else:
                    count += 1
    return mytarget

for u in range(9,10):
    mytar = returntarget(u)
    yo = mytar.split(',')
    newstring = 'name    &' + yo[2] + '   &' + yo[3] + ' [' + yo[4] + ',' + yo[5] + ',' + yo[6] + ']' + '   &'+ yo[7] + ' [' + yo[8] + ',' + yo[9] + ',' + yo[10] + ']' + '   &' + '81 \%     & 81 \%' +'\\'
    print(newstring)

