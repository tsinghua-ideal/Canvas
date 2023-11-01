import os 
import json


length = 40
single_result_folder = f"/scorpio/home/shenao/myProject/Canvas/experiments/collections/validation_experiments/final/{length}/score_new"
subfolders = [f.name for f in os.scandir(single_result_folder) if f.is_dir()]
array1 = [0] * length
array2 = [0] * 2
for f in os.scandir(single_result_folder):
    target_file = os.path.join(single_result_folder, f'{f.name}/metrics.json')
    if os.path.exists(target_file):
        with open(target_file, "r") as json_file:
            data = json.load(json_file)
            if data['eval_metrics'][-1]['top1'] < 10:
                continue
            for i in range(length):
                if (data['extra']['compare_dict_full']['epoch_66'][f'{i + 1}'][1] == 1):
                    array1[i] += 1
                    if i < 0.75 * length:
                        array2[0] += 1
                    print(data['eval_metrics'][-1]['top1'])
                    break
print(len(subfolders))
print(array1)
print(array2)
