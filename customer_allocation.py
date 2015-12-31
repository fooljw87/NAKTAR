from cpmoptimize import cpmoptimize, xrange
import numpy as np
import matplotlib

@cpmoptimize()





################# kth biggest module ################

def kth_biggest(seq, k):
    """ kth_biggest without sorting all array or repeating to find maximum k times
        will crash sooner or later becouse of recursion and Python's recursion limit,
        but not so soon because of divide and conquer algorithm inspired by quicksort
    """
    assert k > 0,'k must be positive %i' % k
    ls = len(seq)
    if ls < k:
        raise ValueError('kth %i: only %i elements' % (k, ls))
    if k == 1:
        return max(seq)
    if k == ls:
        return min(seq)
    pivot = seq[0]
    # others except the pivot compared with pivot
    bigger =  [item for item in seq[1:] if item > pivot]
    #debug
    #print(('seq: %s, k: %i,\npivot: %i,\nbigger: %s\n' %
    #       (seq, k, pivot, bigger)))
    if len(bigger) == k - 1:
        return pivot
    elif len(bigger) >= k:
        return kth_biggest(bigger, k)
    not_bigger = [item for item in seq[1:] if item <= pivot]
    #debug
    #print('not_bigger:', not_bigger)
    return kth_biggest(not_bigger, k - len(bigger) - 1)
def kth_smallest(seq, k):
    """ kth_smallest uses kth_biggest to count kth_smallest """
    return kth_biggest(seq, len(seq) - k + 1)

#####

C000 = 'c000'
C001 = 'c001'
C002 = 'c002'
C003 = 'c003'
C004 = 'c004'
C005 = 'c005'
C006 = 'c006'
C007 = 'c007'
C008 = 'c008'
C009 = 'c009'
C010 = 'c010'
C011 = 'c011'
C012 = 'c012'
C013 = 'c013'
C014 = 'c014'
C015 = 'c015'
C016 = 'c016'
C017 = 'c017'
C018 = 'c018'
C019 = 'c019'

m = 5
n = 20

jbasket = np.array([[1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5]])



zeroset = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])

Superdata = [0]

CAP = [3000, 10000, 15500, 1200, 20000]         # capacity of each factory
                                                # [0] Yeosu
                                                # [1] Ulsan
                                                # [2] Daesan
                                                # [3] Dangjin
                                                # [4] Ansan

buy_price = np.array([3, 110, 135, 1, 220]) # sourcing cost for each factory
                                                # [0] Yeosu
                                                # [1] Ulsan
                                                # [2] Daesan
                                                # [3] Dangjin
                                                # [4] Ansan

cost_per_km = 100

D = np.zeros((20,5), dtype = int)
P = np.zeros((20,5), dtype = float)
weight = np.zeros((20,1), dtype = float)
X = np.zeros((20,5), dtype = int)

remaining_CAP = np.zeros((5,1), dtype = float)

cuslist = []
cusdata = []

class Customer:

    cusnumber = 0

    def __init__(self, name):
        self.name = name
        self.distance = []
        self.price = 0
        self.number = Customer.cusnumber
        cuslist.append((self.number, self.name))
        Customer.cusnumber += 1

    def set_distance(self, distance):
        for x in distance:
            self.distance.append(x)

    def set_price(self, price):
        self.price = price

    def set_weight(self, weight):
        self.weight = weight

    def apply_data(self):
        for i in xrange(0,5):
            P[self.number, i] = self.weight * (self.price - buy_price[i] - (self.distance[i] * cost_per_km))
        cusdata.append((self.number, self.name, self.price, self.weight, self.distance))
        D[self.number] = self.distance
        weight[self.number] = self.weight

#######
C000 = Customer('0')
C000.set_distance([9,8,6,4,1])
C000.set_price(555)
C000.set_weight(300)
C000.apply_data()

C001 = Customer('1')
C001.set_distance([1,3,4,5,8])
C001.set_price(570)
C001.set_weight(900)
C001.apply_data()

C002 = Customer('2')
C002.set_distance([9,2,1,7,6])
C002.set_price(590)
C002.set_weight(300)
C002.apply_data()

C003 = Customer('3')
C003.set_distance([7,5,1,4,9])
C003.set_price(490)
C003.set_weight(200)
C003.apply_data()

C004 = Customer('4')
C004.set_distance([9,5,1,5,4])
C004.set_price(470)
C004.set_weight(700)
C004.apply_data()

C005 = Customer('5')
C005.set_distance([9,2,9,3,1])
C005.set_price(395)
C005.set_weight(1000)
C005.apply_data()

C006 = Customer('6')
C006.set_distance([9,4,5,3,1])
C006.set_price(510)
C006.set_weight(500)
C006.apply_data()

C007 = Customer('7')
C007.set_distance([4,2,4,1,6])
C007.set_price(600)
C007.set_weight(600)
C007.apply_data()

C008 = Customer('8')
C008.set_distance([1,8,7,6,6])
C008.set_price(650)
C008.set_weight(400)
C008.apply_data()

C009 = Customer('9')
C009.set_distance([2,1,8,6,9])
C009.set_price(400)
C009.set_weight(500)
C009.apply_data()

C010 = Customer('10')
C010.set_distance([5,9,5,3,1])
C010.set_price(490)
C010.set_weight(10500)
C010.apply_data()

C011 = Customer('11')
C011.set_distance([4,7,6,5,2])
C011.set_price(410)
C011.set_weight(3000)
C011.apply_data()

C012 = Customer('12')
C012.set_distance([3,3,3,7,6])
C012.set_price(430)
C012.set_weight(250)
C012.apply_data()

C013 = Customer('13')
C013.set_distance([7,1,1,7,2])
C013.set_price(450)
C013.set_weight(270)
C013.apply_data()

C014 = Customer('14')
C014.set_distance([4,2,1,3,2])
C014.set_price(610)
C014.set_weight(550)
C014.apply_data()

C015 = Customer('15')
C015.set_distance([5,5,3,7,1])
C015.set_price(590)
C015.set_weight(1500)
C015.apply_data()

C016 = Customer('16')
C016.set_distance([5,4,8,3,7])
C016.set_price(570)
C016.set_weight(190)
C016.apply_data()

C017 = Customer('17')
C017.set_distance([7,1,3,7,5])
C017.set_price(520)
C017.set_weight(50)
C017.apply_data()

C018 = Customer('18')
C018.set_distance([7,1,3,1,9])
C018.set_price(510)
C018.set_weight(800)
C018.apply_data()

C019 = Customer('19')
C019.set_distance([7,7,2,1,7])
C019.set_price(470)
C019.set_weight(200)
C019.apply_data()


#########

print cuslist
print D
print P

for (a,b,c,d,e) in cusdata:
    if a == 3:
        print b,c,d,e

m = 5
n = 20

total_profit = sum(sum(X*P))

################################################# make pools

in_basket = [True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False]

jlist = []

counter = 0
for j in xrange(0,40):
    if in_basket[j] == True:
        jlist.append(j)

jpool = np.zeros((len(jlist), 5), dtype = int)

#jpool_counter = 0

#while jpool_counter <= 4:
#    for j in xrange(0, len(jlist)):
#for i in xrange(0,5):
#            if i == jpool_counter:
#                jpool[j,i,jpool_counter] = jpool_counter + 1
#                jpool_counter += 1

for j in xrange(0, len(jlist)):
    jpool[j,:] = [1,2,3,4,5]

P_pool = np.zeros((len(jlist),m), dtype = float)

print jpool

for i in xrange(0,m):
    for j in xrange(len(jlist)):
        P_pool[j,i] = P[jlist[j], i]

weight_pool = np.zeros((len(jlist), 1), dtype = float)

for j in xrange(len(jlist)):
    weight_pool[j] = weight[jlist[j]]

X_pool = np.zeros((len(jlist),m), dtype = int)

#############################################################


flow_rate = X * weight

for i in xrange(0,5):
    remaining_CAP[i] = CAP[i] - sum(flow_rate[:,i])

opcost = np.zeros((n,m), dtype = float)
K = np.zeros((n,m), dtype = int)


profit_per_unit = np.zeros((30,5), dtype = float)
Kvert = [[],[],[],[],[]]


for i in xrange(0,m):
    for j in xrange(0,n):
        profit_per_unit[j,i] = P[j,i] / weight[j]


print 'jlist'
print jlist
print 'jpool'
print jpool
print 'P_pool'
print P_pool
print 'CAP'
print CAP
print 'weight_pool'
print weight_pool

K = np.zeros((len(jlist), m), dtype = int)

for j in xrange(0, len(jlist)):
    for i in xrange(0,m):
        for k in xrange(0,5):
            if P_pool[j,i] == kth_biggest(P_pool[j,:], k+1):
                K[j,i] = k + 1

print K

for i in xrange(0,m):
    for j in xrange(0,len(jlist)):
        for k in xrange(0,5):
            if P_pool[j,i] == kth_biggest(P_pool[j,:], k+1):
                jpool[j,k] = i+1


for j in xrange(0, len(jlist)):
    for idx, val in enumerate(jpool[j,:]):
        if CAP[val-1] - weight_pool[j] < 0:
            jpool[j,idx] = 0

print jpool

add_all = len(jlist)
#add_to_counter = 0
#add_to_add_all = 0

#while add_to_counter <= len(jlist):
#    for idx, val in enumerate(jpool[add_to_counter,:]):
#        if val != 0:
#            add_to_add_all += idx
#            add_to_counter += 1


#add_all = add_all + add_to_add_all

print 'add_all'
print add_all

def fun():
    aa0 = 0
    a0 = jpool[0,aa0]
    if len(jlist) == 1:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all: continue
                aa0 += 1
                val_array = np.array([zeroset[a0-1,:]])
                if add_a0 == add_all:
                    if sum(sum(val_array*P_pool)) > max(Superdata):
                        if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                            if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                    if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                        if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                            list.append(Superdata, sum(sum(val_array * P_pool)))
                                            X_pool = val_array
                                            print X_pool
                                            print Superdata
                                            print max(Superdata)
                                            break_counter = 0
                                            starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break

    if len(jlist) == 2:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 1: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all: continue
                    aa1 += 1
                    val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:]])
                    if add_a1 == add_all:
                        if sum(sum(val_array*P_pool)) > max(Superdata):
                            if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                    if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                        if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                            if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                list.append(Superdata, sum(sum(val_array * P_pool)))
                                                X_pool = val_array
                                                print X_pool
                                                print Superdata
                                                print max(Superdata)
                                                break_counter = 0
                                                starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break

    if len(jlist) == 3:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 12: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 1: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all: continue
                        aa2 += 1
                        val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:]])
                        if add_a2 == add_all:
                            if sum(sum(val_array*P_pool)) > max(Superdata):
                                if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                    if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                        if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                            if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                    list.append(Superdata, sum(sum(val_array * P_pool)))
                                                    X_pool = val_array
                                                    print X_pool
                                                    print Superdata
                                                    print max(Superdata)
                                                    break_counter = 0
                                                    starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break

    if len(jlist) == 4:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 3: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 2: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 1: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all: continue
                            aa3 += 1
                            val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:]])
                            if add_a3 == add_all:
                                if sum(sum(val_array*P_pool)) > max(Superdata):
                                    if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                        if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                            if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                    if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                        list.append(Superdata, sum(sum(val_array * P_pool)))
                                                        X_pool = val_array
                                                        print X_pool
                                                        print Superdata
                                                        print max(Superdata)
                                                        break_counter = 0
                                                        starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 5:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 4: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 3: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 2: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 1: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all: continue
                                aa4 += 1
                                val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:]])
                                if add_a4 == add_all:
                                    if sum(sum(val_array*P_pool)) > max(Superdata):
                                        if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                            if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                    if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                        if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                            list.append(Superdata, sum(sum(val_array * P_pool)))
                                                            X_pool = val_array
                                                            print X_pool
                                                            print Superdata
                                                            print max(Superdata)
                                                            break_counter = 0
                                                            starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 6:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 5: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 4: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 3: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 2: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 1: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all: continue
                                    aa5 += 1
                                    val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                    zeroset[a5-1,:]])
                                    if add_a5 == add_all:
                                        if sum(sum(val_array*P_pool)) > max(Superdata):
                                            if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                    if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                        if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                            if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                X_pool = val_array
                                                                print X_pool
                                                                print Superdata
                                                                print max(Superdata)
                                                                break_counter = 0
                                                                starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 7:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 6: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 5: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 4: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 3: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 2: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 1: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all: continue
                                        aa6 += 1
                                        val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                        zeroset[a5-1,:],zeroset[a6-1,:]])
                                        if add_a6 == add_all:
                                            if sum(sum(val_array*P_pool)) > max(Superdata):
                                                if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                    if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                        if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                            if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                    list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                    X_pool = val_array
                                                                    print X_pool
                                                                    print Superdata
                                                                    print max(Superdata)
                                                                    break_counter = 0
                                                                    starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 8:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 7: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 6: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 5: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 4: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 3: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 2: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 1: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all: continue
                                            aa7 += 1
                                            val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                            zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:]])
                                            if add_a7 == add_all:
                                                if sum(sum(val_array*P_pool)) > max(Superdata):
                                                    if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                        if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                            if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                    if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                        list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                        X_pool = val_array
                                                                        print X_pool
                                                                        print Superdata
                                                                        print max(Superdata)
                                                                        break_counter = 0
                                                                        starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 9:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 8: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 7: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 6: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 5: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 4: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 3: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 2: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 1: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all: continue
                                                aa8 += 1
                                                val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:]])
                                                if add_a8 == add_all:
                                                    if sum(sum(val_array*P_pool)) > max(Superdata):
                                                        if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                            if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                    if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                        if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                            list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                            X_pool = val_array
                                                                            print X_pool
                                                                            print Superdata
                                                                            print max(Superdata)
                                                                            break_counter = 0
                                                                            starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break



    if len(jlist) == 10:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 9: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 8: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 7: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 6: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 5: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 4: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 3: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 2: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 1: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all: continue
                                                    aa9 += 1
                                                    val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                    zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:]])
                                                    if add_a9 == add_all:
                                                        if sum(sum(val_array*P_pool)) > max(Superdata):
                                                            if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                    if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                        if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                            if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                X_pool = val_array
                                                                                print X_pool
                                                                                print Superdata
                                                                                print max(Superdata)
                                                                                break_counter = 0
                                                                                starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 11:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 10: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 9: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 8: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 7: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 6: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 5: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 4: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 3: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 2: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all - 1: continue
                                                    aa9 += 1
                                                    aa10 = 0
                                                    a10 = jpool[10,aa10]
                                                    for a10 in jpool[10,:]:
                                                        if a10 == 0: continue
                                                        add_a10 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1]
                                                        if add_a10 > add_all: continue
                                                        val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                        zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:],
                                                                        zeroset[a10-1,:]])
                                                        if add_a10 == add_all:
                                                            if sum(sum(val_array*P_pool)) > max(Superdata):
                                                                if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                    if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                        if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                            if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                                if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                    list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                    X_pool = val_array
                                                                                    print X_pool
                                                                                    print Superdata
                                                                                    print max(Superdata)
                                                                                    break_counter = 0
                                                                                    starting_condition = True
                                                        aa10 += 1
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 12:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 11: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 10: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 9: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 8: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 7: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 6: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 5: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 4: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 3: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all - 2: continue
                                                    aa9 += 1
                                                    aa10 = 0
                                                    a10 = jpool[10,aa10]
                                                    for a10 in jpool[10,:]:
                                                        if a10 == 0: continue
                                                        add_a10 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1]
                                                        if add_a10 > add_all - 1: continue
                                                        aa10 += 1
                                                        aa11 = 0
                                                        a11 = jpool[11,aa11]
                                                        for a11 in jpool[11,:]:
                                                            if a11 == 0: continue
                                                            add_a11 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1]
                                                            if add_a11 > add_all: continue
                                                            aa11 += 1
                                                            val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                            zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:],
                                                                            zeroset[a10-1,:],zeroset[a11-1,:]])
                                                            if add_a11 == add_all:
                                                                if sum(sum(val_array*P_pool)) > max(Superdata):
                                                                    if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                        if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                            if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                                if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                                    if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                        list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                        X_pool = val_array
                                                                                        print X_pool
                                                                                        print Superdata
                                                                                        print max(Superdata)
                                                                                        break_counter = 0
                                                                                        starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 13:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 12: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 11: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 10: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 9: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 8: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 7: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 6: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 5: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 4: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all - 3: continue
                                                    aa9 += 1
                                                    aa10 = 0
                                                    a10 = jpool[10,aa10]
                                                    for a10 in jpool[10,:]:
                                                        if a10 == 0: continue
                                                        add_a10 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1]
                                                        if add_a10 > add_all - 2: continue
                                                        aa10 += 1
                                                        aa11 = 0
                                                        a11 = jpool[11,aa11]
                                                        for a11 in jpool[11,:]:
                                                            if a11 == 0: continue
                                                            add_a11 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1]
                                                            if add_a11 > add_all - 1: continue
                                                            aa11 += 1
                                                            aa12 = 0
                                                            a12 = jpool[12,aa12]
                                                            for a12 in jpool[12,:]:
                                                                if a12 == 0: continue
                                                                add_a12 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1]
                                                                if add_a12 > add_all: continue
                                                                aa12 += 1
                                                                val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                                zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:],
                                                                                zeroset[a10-1,:],zeroset[a11-1,:],zeroset[a12-1,:]])
                                                                if add_a12 == add_all:
                                                                    if sum(sum(val_array*P_pool)) > max(Superdata):
                                                                        if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                            if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                                if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                                    if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                                        if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                            list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                            X_pool = val_array
                                                                                            print X_pool
                                                                                            print Superdata
                                                                                            print max(Superdata)
                                                                                            break_counter = 0
                                                                                            starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break



    if len(jlist) == 14:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 13: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 12: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 11: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 10: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 9: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 8: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 7: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 6: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 5: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all - 4: continue
                                                    aa9 += 1
                                                    aa10 = 0
                                                    a10 = jpool[10,aa10]
                                                    for a10 in jpool[10,:]:
                                                        if a10 == 0: continue
                                                        add_a10 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1]
                                                        if add_a10 > add_all - 3: continue
                                                        aa10 += 1
                                                        aa11 = 0
                                                        a11 = jpool[11,aa11]
                                                        for a11 in jpool[11,:]:
                                                            if a11 == 0: continue
                                                            add_a11 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1]
                                                            if add_a11 > add_all - 2: continue
                                                            aa11 += 1
                                                            aa12 = 0
                                                            a12 = jpool[12,aa12]
                                                            for a12 in jpool[12,:]:
                                                                if a12 == 0: continue
                                                                add_a12 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1]
                                                                if add_a12 > add_all - 1: continue
                                                                aa12 += 1
                                                                aa13 = 0
                                                                a13 = jpool[9,aa13]
                                                                for a13 in jpool[8,:]:
                                                                    if a13 == 0: continue
                                                                    add_a13 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1]
                                                                    if add_a13 > add_all: continue
                                                                    aa13 += 1
                                                                    val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                                    zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:],
                                                                                    zeroset[a10-1,:],zeroset[a11-1,:],zeroset[a12-1,:],zeroset[a13-1,:]])
                                                                    if add_a13 == add_all:
                                                                        if sum(sum(val_array*P_pool)) > max(Superdata):
                                                                            if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                                if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                                    if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                                        if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                                            if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                                list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                                X_pool = val_array
                                                                                                print X_pool
                                                                                                print Superdata
                                                                                                print max(Superdata)
                                                                                                break_counter = 0
                                                                                                starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break

    if len(jlist) == 15:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 14: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 13: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 12: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 11: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 10: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 9: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 8: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 7: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 6: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all - 5: continue
                                                    aa9 += 1
                                                    aa10 = 0
                                                    a10 = jpool[10,aa10]
                                                    for a10 in jpool[10,:]:
                                                        if a10 == 0: continue
                                                        add_a10 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1]
                                                        if add_a10 > add_all - 4: continue
                                                        aa10 += 1
                                                        aa11 = 0
                                                        a11 = jpool[11,aa11]
                                                        for a11 in jpool[11,:]:
                                                            if a11 == 0: continue
                                                            add_a11 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1]
                                                            if add_a11 > add_all - 3: continue
                                                            aa11 += 1
                                                            aa12 = 0
                                                            a12 = jpool[12,aa12]
                                                            for a12 in jpool[12,:]:
                                                                if a12 == 0: continue
                                                                add_a12 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1]
                                                                if add_a12 > add_all - 2: continue
                                                                aa12 += 1
                                                                aa13 = 0
                                                                a13 = jpool[9,aa13]
                                                                for a13 in jpool[8,:]:
                                                                    if a13 == 0: continue
                                                                    add_a13 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1]
                                                                    if add_a13 > add_all - 1: continue
                                                                    aa13 += 1
                                                                    aa14 = 0
                                                                    a14 = jpool[14,aa14]
                                                                    for a14 in jpool[14,:]:
                                                                        if a14 == 0: continue
                                                                        add_a14 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1]
                                                                        if add_a14 > add_all: continue
                                                                        aa14 += 1
                                                                        val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                                        zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:],
                                                                                        zeroset[a10-1,:],zeroset[a11-1,:],zeroset[a12-1,:],zeroset[a13-1,:],zeroset[a14-1,:]])
                                                                        if add_a14 == add_all:
                                                                            if sum(sum(val_array*P_pool)) > max(Superdata):
                                                                                if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                                    if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                                        if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                                            if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                                                if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                                    list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                                    X_pool = val_array
                                                                                                    print X_pool
                                                                                                    print Superdata
                                                                                                    print max(Superdata)
                                                                                                    break_counter = 0
                                                                                                    starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 16:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 15: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 14: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 13: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 12: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 11: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 10: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 9: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 8: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 7: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all - 6: continue
                                                    aa9 += 1
                                                    aa10 = 0
                                                    a10 = jpool[10,aa10]
                                                    for a10 in jpool[10,:]:
                                                        if a10 == 0: continue
                                                        add_a10 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1]
                                                        if add_a10 > add_all - 5: continue
                                                        aa10 += 1
                                                        aa11 = 0
                                                        a11 = jpool[11,aa11]
                                                        for a11 in jpool[11,:]:
                                                            if a11 == 0: continue
                                                            add_a11 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1]
                                                            if add_a11 > add_all - 4: continue
                                                            aa11 += 1
                                                            aa12 = 0
                                                            a12 = jpool[12,aa12]
                                                            for a12 in jpool[12,:]:
                                                                if a12 == 0: continue
                                                                add_a12 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1]
                                                                if add_a12 > add_all - 3: continue
                                                                aa12 += 1
                                                                aa13 = 0
                                                                a13 = jpool[9,aa13]
                                                                for a13 in jpool[8,:]:
                                                                    if a13 == 0: continue
                                                                    add_a13 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1]
                                                                    if add_a13 > add_all - 2: continue
                                                                    aa13 += 1
                                                                    aa14 = 0
                                                                    a14 = jpool[14,aa14]
                                                                    for a14 in jpool[14,:]:
                                                                        if a14 == 0: continue
                                                                        add_a14 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1]
                                                                        if add_a14 > add_all - 1: continue
                                                                        aa14 += 1
                                                                        aa15 = 0
                                                                        a15 = jpool[15,aa15]
                                                                        for a15 in jpool[15,:]:
                                                                            if a15 == 0: continue
                                                                            add_a15 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1]
                                                                            if add_a15 > add_all: continue
                                                                            aa15 += 1
                                                                            val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                                            zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:],
                                                                                            zeroset[a10-1,:],zeroset[a11-1,:],zeroset[a12-1,:],zeroset[a13-1,:],zeroset[a14-1,:],
                                                                                            zeroset[a15-1,:]])
                                                                            if add_a15 == add_all:
                                                                                if sum(sum(val_array*P_pool)) > max(Superdata):
                                                                                    if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                                        if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                                            if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                                                if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                                                    if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                                        list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                                        X_pool = val_array
                                                                                                        print X_pool
                                                                                                        print Superdata
                                                                                                        print max(Superdata)
                                                                                                        break_counter = 0
                                                                                                        starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 17:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 16: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 15: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 14: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 13: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 12: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 11: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 10: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 9: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 8: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all - 7: continue
                                                    aa9 += 1
                                                    aa10 = 0
                                                    a10 = jpool[10,aa10]
                                                    for a10 in jpool[10,:]:
                                                        if a10 == 0: continue
                                                        add_a10 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1]
                                                        if add_a10 > add_all - 6: continue
                                                        aa10 += 1
                                                        aa11 = 0
                                                        a11 = jpool[11,aa11]
                                                        for a11 in jpool[11,:]:
                                                            if a11 == 0: continue
                                                            add_a11 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1]
                                                            if add_a11 > add_all - 5: continue
                                                            aa11 += 1
                                                            aa12 = 0
                                                            a12 = jpool[12,aa12]
                                                            for a12 in jpool[12,:]:
                                                                if a12 == 0: continue
                                                                add_a12 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1]
                                                                if add_a12 > add_all - 4: continue
                                                                aa12 += 1
                                                                aa13 = 0
                                                                a13 = jpool[9,aa13]
                                                                for a13 in jpool[8,:]:
                                                                    if a13 == 0: continue
                                                                    add_a13 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1]
                                                                    if add_a13 > add_all - 3: continue
                                                                    aa13 += 1
                                                                    aa14 = 0
                                                                    a14 = jpool[14,aa14]
                                                                    for a14 in jpool[14,:]:
                                                                        if a14 == 0: continue
                                                                        add_a14 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1]
                                                                        if add_a14 > add_all - 2: continue
                                                                        aa14 += 1
                                                                        aa15 = 0
                                                                        a15 = jpool[15,aa15]
                                                                        for a15 in jpool[15,:]:
                                                                            if a15 == 0: continue
                                                                            add_a15 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1]
                                                                            if add_a15 > add_all - 1: continue
                                                                            aa15 += 1
                                                                            aa16 = 0
                                                                            a16 = jpool[16,aa16]
                                                                            for a16 in jpool[16,:]:
                                                                                if a16 == 0: continue
                                                                                add_a16 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1] + K[16,a16-1]
                                                                                if add_a16 > add_all: continue
                                                                                aa16 += 1
                                                                                val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                                                zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:],
                                                                                                zeroset[a10-1,:],zeroset[a11-1,:],zeroset[a12-1,:],zeroset[a13-1,:],zeroset[a14-1,:],
                                                                                                zeroset[a15-1,:],zeroset[a16-1,:]])
                                                                                if add_a16 == add_all:
                                                                                    if sum(sum(val_array*P_pool)) > max(Superdata):
                                                                                        if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                                            if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                                                if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                                                    if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                                                        if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                                            list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                                            X_pool = val_array
                                                                                                            print X_pool
                                                                                                            print Superdata
                                                                                                            print max(Superdata)
                                                                                                            break_counter = 0
                                                                                                            starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break



    if len(jlist) == 18:
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 17: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 16: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 15: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 14: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 13: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 12: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 11: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 10: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 9: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all - 8: continue
                                                    aa9 += 1
                                                    aa10 = 0
                                                    a10 = jpool[10,aa10]
                                                    for a10 in jpool[10,:]:
                                                        if a10 == 0: continue
                                                        add_a10 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1]
                                                        if add_a10 > add_all - 7: continue
                                                        aa10 += 1
                                                        aa11 = 0
                                                        a11 = jpool[11,aa11]
                                                        for a11 in jpool[11,:]:
                                                            if a11 == 0: continue
                                                            add_a11 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1]
                                                            if add_a11 > add_all - 6: continue
                                                            aa11 += 1
                                                            aa12 = 0
                                                            a12 = jpool[12,aa12]
                                                            for a12 in jpool[12,:]:
                                                                if a12 == 0: continue
                                                                add_a12 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1]
                                                                if add_a12 > add_all - 5: continue
                                                                aa12 += 1
                                                                aa13 = 0
                                                                a13 = jpool[9,aa13]
                                                                for a13 in jpool[8,:]:
                                                                    if a13 == 0: continue
                                                                    add_a13 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1]
                                                                    if add_a13 > add_all - 4: continue
                                                                    aa13 += 1
                                                                    aa14 = 0
                                                                    a14 = jpool[14,aa14]
                                                                    for a14 in jpool[14,:]:
                                                                        if a14 == 0: continue
                                                                        add_a14 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1]
                                                                        if add_a14 > add_all - 3: continue
                                                                        aa14 += 1
                                                                        aa15 = 0
                                                                        a15 = jpool[15,aa15]
                                                                        for a15 in jpool[15,:]:
                                                                            if a15 == 0: continue
                                                                            add_a15 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1]
                                                                            if add_a15 > add_all - 2: continue
                                                                            aa15 += 1
                                                                            aa16 = 0
                                                                            a16 = jpool[16,aa16]
                                                                            for a16 in jpool[16,:]:
                                                                                if a16 == 0: continue
                                                                                add_a16 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1] + K[16,a16-1]
                                                                                if add_a16 > add_all - 1: continue
                                                                                aa16 += 1
                                                                                aa17 = 0
                                                                                a17 = jpool[17,aa17]
                                                                                for a17 in jpool[17,:]:
                                                                                    if a17 == 0: continue
                                                                                    add_a17 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1] + K[16,a16-1] + K[17,a17-1]
                                                                                    if add_a17 > add_all: continue
                                                                                    aa17 += 1
                                                                                    val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                                                    zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:],
                                                                                                    zeroset[a10-1,:],zeroset[a11-1,:],zeroset[a12-1,:],zeroset[a13-1,:],zeroset[a14-1,:],
                                                                                                    zeroset[a15-1,:],zeroset[a16-1,:],zeroset[a17-1,:]])
                                                                                    if add_a17 == add_all:
                                                                                        if sum(sum(val_array*P_pool)) > max(Superdata):
                                                                                            if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                                                if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                                                    if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                                                        if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                                                            if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                                                list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                                                X_pool = val_array
                                                                                                                print X_pool
                                                                                                                print Superdata
                                                                                                                print max(Superdata)
                                                                                                                break_counter = 0
                                                                                                                starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break


    if len(jlist) == 19:
        add_all = len(jlist)
        val_array = np.zeros((len(jlist), m), dtype = int)
        break_counter = 0
        add_all = len(jlist)
        starting_condition = False
        while add_all <= len(jlist) * 5:
            for a0 in jpool[0,:]:
                if a0 == 0: continue
                add_a0 = K[0,a0-1]
                if add_a0 > add_all - 18: continue
                aa0 += 1
                aa1 = 0
                a1 = jpool[1,aa1]
                for a1 in jpool[1,:]:
                    if a1 == 0: continue
                    add_a1 = K[0,a0-1] + K[1,a1-1]
                    if add_a1 > add_all - 17: continue
                    aa1 += 1
                    aa2 = 0
                    a2 = jpool[2,aa2]
                    for a2 in jpool[2,:]:
                        if a2 == 0: continue
                        add_a2 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1]
                        if add_a2 > add_all - 16: continue
                        aa2 += 1
                        aa3 = 0
                        a3 = jpool[3,aa3]
                        for a3 in jpool[3,:]:
                            if a3 == 0: continue
                            add_a3 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1]
                            if add_a3 > add_all - 15: continue
                            aa3 += 1
                            aa4 = 0
                            a4 = jpool[4,aa4]
                            for a4 in jpool[4,:]:
                                if a4 == 0: continue
                                add_a4 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1]
                                if add_a4 > add_all - 14: continue
                                aa4 += 1
                                aa5 = 0
                                a5 = jpool[5,aa5]
                                for a5 in jpool[5,:]:
                                    if a5 == 0: continue
                                    add_a5 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1]
                                    if add_a5 > add_all - 13: continue
                                    aa5 += 1
                                    aa6 = 0
                                    a6 = jpool[6,aa6]
                                    for a6 in jpool[6,:]:
                                        if a6 == 0: continue
                                        add_a6 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1]
                                        if add_a6 > add_all - 12: continue
                                        aa6 += 1
                                        aa7 = 0
                                        a7 = jpool[7,aa7]
                                        for a7 in jpool[7,:]:
                                            if a7 == 0: continue
                                            add_a7 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1]
                                            if add_a7 > add_all - 11: continue
                                            aa7 += 1
                                            aa8 = 0
                                            a8 = jpool[8,aa8]
                                            for a8 in jpool[8,:]:
                                                if a8 == 0: continue
                                                add_a8 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1]
                                                if add_a8 > add_all - 10: continue
                                                aa8 += 1
                                                aa9 = 0
                                                a9 = jpool[9,aa9]
                                                for a9 in jpool[9,:]:
                                                    if a9 == 0: continue
                                                    add_a9 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1]
                                                    if add_a9 > add_all - 9: continue
                                                    aa9 += 1
                                                    aa10 = 0
                                                    a10 = jpool[10,aa10]
                                                    for a10 in jpool[10,:]:
                                                        if a10 == 0: continue
                                                        add_a10 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1]
                                                        if add_a10 > add_all - 8: continue
                                                        aa10 += 1
                                                        aa11 = 0
                                                        a11 = jpool[11,aa11]
                                                        for a11 in jpool[11,:]:
                                                            if a11 == 0: continue
                                                            add_a11 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1]
                                                            if add_a11 > add_all - 7: continue
                                                            aa11 += 1
                                                            aa12 = 0
                                                            a12 = jpool[12,aa12]
                                                            for a12 in jpool[12,:]:
                                                                if a12 == 0: continue
                                                                add_a12 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1]
                                                                if add_a12 > add_all - 6: continue
                                                                aa12 += 1
                                                                aa13 = 0
                                                                a13 = jpool[9,aa13]
                                                                for a13 in jpool[8,:]:
                                                                    if a13 == 0: continue
                                                                    add_a13 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1]
                                                                    if add_a13 > add_all - 5: continue
                                                                    aa13 += 1
                                                                    aa14 = 0
                                                                    a14 = jpool[14,aa14]
                                                                    for a14 in jpool[14,:]:
                                                                        if a14 == 0: continue
                                                                        add_a14 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1]
                                                                        if add_a14 > add_all - 4: continue
                                                                        aa14 += 1
                                                                        aa15 = 0
                                                                        a15 = jpool[15,aa15]
                                                                        for a15 in jpool[15,:]:
                                                                            if a15 == 0: continue
                                                                            add_a15 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1]
                                                                            if add_a15 > add_all - 3: continue
                                                                            aa15 += 1
                                                                            aa16 = 0
                                                                            a16 = jpool[16,aa16]
                                                                            for a16 in jpool[16,:]:
                                                                                if a16 == 0: continue
                                                                                add_a16 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1] + K[16,a16-1]
                                                                                if add_a16 > add_all - 2: continue
                                                                                aa16 += 1
                                                                                aa17 = 0
                                                                                a17 = jpool[17,aa17]
                                                                                for a17 in jpool[17,:]:
                                                                                    if a17 == 0: continue
                                                                                    add_a17 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1] + K[16,a16-1] + K[17,a17-1]
                                                                                    if add_a17 > add_all - 1: continue
                                                                                    aa17 += 1
                                                                                    aa18 = 0
                                                                                    a18 = jpool[18,aa18]
                                                                                    for a18 in jpool[18,:]:
                                                                                        if a18 == 0: continue
                                                                                        add_a18 = K[0,a0-1] + K[1,a1-1] + K[2,a2-1] + K[3,a3-1] + K[4,a4-1] + K[5,a5-1] + K[6,a6-1] + K[7,a7-1] + K[8,a8-1] + K[9,a9-1] + K[10,a10-1] + K[11,a11-1] + K[12,a12-1] + K[13,a13-1] + K[14,a14-1] + K[15,a15 -1] + K[16,a16-1] + K[17,a17-1] + K[18,a18-1]
                                                                                        if add_a18 > add_all: continue
                                                                                        aa18 += 1
                                                                                        val_array = np.array([zeroset[a0-1,:],zeroset[a1-1,:],zeroset[a2-1,:],zeroset[a3-1,:],zeroset[a4-1,:],
                                                                                                        zeroset[a5-1,:],zeroset[a6-1,:],zeroset[a7-1,:], zeroset[a8-1,:],zeroset[a9-1,:],
                                                                                                        zeroset[a10-1,:],zeroset[a11-1,:],zeroset[a12-1,:],zeroset[a13-1,:],zeroset[a14-1,:],
                                                                                                        zeroset[a15-1,:],zeroset[a16-1,:],zeroset[a17-1,:], zeroset[a18-1,:]])
                                                                                        if add_a18 == add_all:
                                                                                            if sum(sum(val_array*P_pool)) > max(Superdata):
                                                                                                if CAP[0] >= sum((val_array * weight_pool)[:,0]):
                                                                                                    if CAP[1] >= sum((val_array * weight_pool)[:,1]):
                                                                                                        if CAP[2] >= sum((val_array * weight_pool)[:,2]):
                                                                                                            if CAP[3] >= sum((val_array * weight_pool)[:,3]):
                                                                                                                if CAP[4] >= sum((val_array * weight_pool)[:,4]):
                                                                                                                    list.append(Superdata, sum(sum(val_array * P_pool)))
                                                                                                                    X_pool = val_array
                                                                                                                    print X_pool
                                                                                                                    print Superdata
                                                                                                                    print max(Superdata)
                                                                                                                    break_counter = 0
                                                                                                                    starting_condition = True
            add_all += 1
            print 'add_all'
            print add_all
            if starting_condition == True:
                break_counter += 1
            if break_counter == 4:
                break

fun()
