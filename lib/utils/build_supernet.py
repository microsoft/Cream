# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

def search_for_layer(flops_op_dict, arch_def, flops_minimum, flops_maximum):
    sta_num = [1, 1, 1, 1, 1]
    order = [2, 3, 4, 1, 0, 2, 3, 4, 1, 0]
    limits = [3, 3, 3, 2, 2, 4, 4, 4, 4, 4]
    size_factor = 7
    base_min_flops = sum([flops_op_dict[i][0][0] for i in range(5)])
    base_max_flops = sum([flops_op_dict[i][5][0] for i in range(5)])

    if base_min_flops > flops_maximum:
        while base_min_flops > flops_maximum and size_factor >= 2:
            size_factor = size_factor - 1
            flops_minimum = flops_minimum * (7. / size_factor)
            flops_maximum = flops_maximum * (7. / size_factor)
        if size_factor < 2:
            return None, None, None
    elif base_max_flops < flops_minimum:
        cur_ptr = 0
        while base_max_flops < flops_minimum and cur_ptr <= 9:
            if sta_num[order[cur_ptr]] >= limits[cur_ptr]:
                cur_ptr += 1
                continue
            base_max_flops = base_max_flops + \
                flops_op_dict[order[cur_ptr]][5][1]
            sta_num[order[cur_ptr]] += 1
        if cur_ptr > 7 and base_max_flops < flops_minimum:
            return None, None, None

    cur_ptr = 0
    while cur_ptr <= 9:
        if sta_num[order[cur_ptr]] >= limits[cur_ptr]:
            cur_ptr += 1
            continue
        base_max_flops = base_max_flops + flops_op_dict[order[cur_ptr]][5][1]
        if base_max_flops <= flops_maximum:
            sta_num[order[cur_ptr]] += 1
        else:
            break

    arch_def = [item[:i] for i, item in zip([1] + sta_num + [1], arch_def)]
    # print(arch_def)

    return sta_num, arch_def, size_factor


# what is 5*6*2
# standalone file
flops_op_dict = {}
for i in range(5):
    flops_op_dict[i] = {}
flops_op_dict[0][0] = (21.828704, 18.820752)
flops_op_dict[0][1] = (32.669328, 28.16048)
flops_op_dict[0][2] = (25.039968, 23.637648)
flops_op_dict[0][3] = (37.486224, 35.385824)
flops_op_dict[0][4] = (29.856864, 30.862992)
flops_op_dict[0][5] = (44.711568, 46.22384)
flops_op_dict[1][0] = (11.808656, 11.86712)
flops_op_dict[1][1] = (17.68624, 17.780848)
flops_op_dict[1][2] = (13.01288, 13.87416)
flops_op_dict[1][3] = (19.492576, 20.791408)
flops_op_dict[1][4] = (14.819216, 16.88472)
flops_op_dict[1][5] = (22.20208, 25.307248)
flops_op_dict[2][0] = (8.198, 10.99632)
flops_op_dict[2][1] = (12.292848, 16.5172)
flops_op_dict[2][2] = (8.69976, 11.99984)
flops_op_dict[2][3] = (13.045488, 18.02248)
flops_op_dict[2][4] = (9.4524, 13.50512)
flops_op_dict[2][5] = (14.174448, 20.2804)
flops_op_dict[3][0] = (12.006112, 15.61632)
flops_op_dict[3][1] = (18.028752, 23.46096)
flops_op_dict[3][2] = (13.009632, 16.820544)
flops_op_dict[3][3] = (19.534032, 25.267296)
flops_op_dict[3][4] = (14.514912, 18.62688)
flops_op_dict[3][5] = (21.791952, 27.9768)
flops_op_dict[4][0] = (11.307456, 15.292416)
flops_op_dict[4][1] = (17.007072, 23.1504)
flops_op_dict[4][2] = (11.608512, 15.894528)
flops_op_dict[4][3] = (17.458656, 24.053568)
flops_op_dict[4][4] = (12.060096, 16.797696)
flops_op_dict[4][5] = (18.136032, 25.40832)
