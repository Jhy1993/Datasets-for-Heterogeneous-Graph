import os
from collections import Counter


def load(fp, sp=',', return_cnt=False, select_cate=False):
    edge = []
    a = set()
    b = set()
    b_list = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(sp)
            if select_cate:
                if line[1] in ['0', '1']:
                    edge.append(line)
                    a.add(line[0])
                    b.add(line[1])
                    b_list.append(line[1])
            else:
                edge.append(line)
                a.add(line[0])
                b.add(line[1])
                b_list.append(line[1])
    print(len(edge), len(a), len(b))
    cnt = Counter(b_list)
    if return_cnt:
        return edge, a, b, cnt
    else:
        return edge, a, b


item_brand, item_set_1, brand_set_1 = load('item_brand.dat')
item_cate, item_set_2, cate_set_1, cate_cnt = load('item_category.dat', return_cnt=True,
                                                   select_cate=True)
# item_view, item_set_3, view_set_1 = load('item_view.dat')
user_item, user_set_1, item_set_4 = load('user_item.dat', sp='\t')
# 2753 2753 334
# 5508 2753 22
# 5694 1435 3857
# 195791 6170 2753
# item-view 的先不用了

item_set = item_set_1 & item_set_2 & item_set_4  # & item_set_4
# 看看 cate 的什么样子的, cate 为0的2731, 为1的2310,就用这俩当label
#  !!! 问题是很多节点同时有0,1的label...

#   选择cate为0,1的item 得到2731个item
# 根据item来重新过滤各种边


def load2(fp, sp=',', item_set=None, item_pos=0, select_cate=False):
    edge = []
    a = set()
    b = set()
    b_list = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(sp)
            if line[item_pos] in item_set:
                if select_cate:
                    if line[1] in ['0', '1']:
                        edge.append(line)
                        a.add(line[0])
                        b.add(line[1])
                        b_list.append(line[1])
                else:
                    edge.append(line)
                    a.add(line[0])
                    b.add(line[1])
                    b_list.append(line[1])

    print(len(edge), len(a), len(b))
    cnt = Counter(b_list)
    return edge, a, b


# item_brand, item_set_1, brand_set_1 = load2(
#     'item_brand.dat', item_set=item_set)
# new_item_cate, new_item_set_2, new_cate_set_1 = load2(
#     'item_category.dat', item_set=item_set, select_cate=True)
# # item_view, item_set_3, view_set_1 = load2('item_view.dat', item_set=item_set)
# user_item, user_set_1, item_set_4 = load2(
#     'user_item.dat', sp='\t', item_set=item_set, item_pos=1)
