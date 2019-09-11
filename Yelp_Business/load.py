"""
user: UB, UU,
business: BU, B-cate-B, B-city-B

files:
user_business.dat, a: 16239, b: 14284, a-b: 198397
user_user.dat, a: 10580, b: 10580, a-b: 158590
business_city, a: 14267, b: 47, a-b: 14267
business_category, a: 14180, b: 511, a-b: 40009

"""
import os
import scipy.io as sio


def load_edge(fp):
    a_list = []
    b_list = []
    ab_list = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            # print(line)
            a_list.append(line[0])
            b_list.append(line[1])
            ab_list.append(line[:2])

    print(f"a: {len(set(a_list))}, b: {len(set(b_list))}, a-b: {len(ab_list)}")
    return set(a_list), set(b_list)


Bus_set_1, Cate_set = load_edge('business_category.dat')
Bus_set_2, City_set = load_edge('business_city.dat')
User_set_1, Bus_set_3 = load_edge('user_business.dat')
User_set_2, User_set_3 = load_edge('user_user.dat')

final_bus_set = Bus_set_3 & Bus_set_2 & Bus_set_1
final_user_set = User_set_1 & User_set_2 & User_set_3
print(len(final_user_set), len(final_bus_set))

# 重新编号
User2ID = {}
ID2User = {}
Bus2ID = {}
for i in list(final_user_set):
    User2ID[i] = len(User2ID)
    
for i in list(final_bus_set):
    Bus2ID[i] = len(Bus2ID)


Cate2ID = {}
for i in list(Cate_set):
    Cate2ID[i] = len(Cate2ID)
    
City2ID = {}
for i in list(City_set):
    City2ID[i] = len(City2ID)


# 根据 两个编号dict来重新梳理数据.
def refine(fp, d1, d2):
    res = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            # print(line)
            if line[0] in d1 and line[1] in d2:
                res.append([d1[line[0]], d2[line[1]]])
    return res, len(res), len(res)/(len(d1)*len(d2))
new_Bus_Cate, n1,sp1 = refine('business_category.dat', Bus2ID, Cate2ID)
new_Bus_City, n2,sp2 = refine('business_city.dat', Bus2ID, City2ID)
new_User_Bus, n3,sp3 = refine('user_business.dat', User2ID, Bus2ID)
new_User_User, n4,sp4 = refine('user_user.dat', User2ID, User2ID)

# !!!!!!!!! 注意这里只是初步筛选,
# TODO 因为if line[0] in d1 and line[1] in d2: 可能会滤除一些user和bus...
# 
#
#f = open('user_user.txt', 'w')
#for line in new_User_User:
#    f.write(str(line[0]))
#    f.write(' ')
#    f.write(str(line[1]))
#    f.write('\n')
#








