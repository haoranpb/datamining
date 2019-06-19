import csv
import random
import numpy as np
import utm

def check_validation(value_dict):
    validate_list = []
    validate_data = {
        # 'Latitude': value_dict['Latitude'],
        # 'Longitude': value_dict['Longitude'],
        'MRTime': value_dict['MRTime'],
        'TrajID': value_dict['TrajID']
    }
    # if float(value_dict['Accuracy']) > 30:
    #     return False, validate_data

    for i in range(6):
        if value_dict['RNCID_' + str(i+1)] != '-999' and value_dict['CellID_' + str(i+1)] != '-999' and value_dict['Dbm_' + str(i+1)] != '-999' \
            and value_dict['AsuLevel_' + str(i+1)] != '-1' and value_dict['SignalLevel_' + str(i+1)] != '-1':
            
            validate_list.append(i)
            validate_data['RNCID_' + str(i+1)] = value_dict['RNCID_' + str(i+1)]
            validate_data['CellID_' + str(i+1)] = value_dict['CellID_' + str(i+1)]
            validate_data['Dbm_' + str(i+1)] = value_dict['Dbm_' + str(i+1)]
            validate_data['AsuLevel_' + str(i+1)] = value_dict['AsuLevel_' + str(i+1)]
            validate_data['SignalLevel_' + str(i+1)] = value_dict['SignalLevel_' + str(i+1)]

    if len(validate_list) < 3: # 我认为基站数量少于3，即理论上不可能得出手机坐标
        return False, validate_data
    else:
        for i in range(6):
            if i not in validate_list:
                np.random.seed(1)
                k = np.random.choice(validate_list)
                validate_data['RNCID_' + str(i+1)] = value_dict['RNCID_' + str(k+1)]
                validate_data['CellID_' + str(i+1)] = value_dict['CellID_' + str(k+1)]
                validate_data['Dbm_' + str(i+1)] = value_dict['Dbm_' + str(k+1)]
                validate_data['AsuLevel_' + str(i+1)] = value_dict['AsuLevel_' + str(k+1)]
                validate_data['SignalLevel_' + str(i+1)] = value_dict['SignalLevel_' + str(k+1)]
        return True, validate_data


group_by_trajid_dict = {}
with open('../data/test_2g.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        validation, parsed_row = check_validation(row)
        if(not validation):
            continue
        traj_id = parsed_row['TrajID']
        if traj_id in group_by_trajid_dict:
            group_by_trajid_dict[traj_id] += 1
        else:
            group_by_trajid_dict[traj_id] = 1

import json
print(json.dumps(group_by_trajid_dict, indent=2))