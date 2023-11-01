import os 
import json
length = 40

single_result_folder = f"/scorpio/home/shenao/myProject/Canvas/experiments/collections/validation_experiments/final/{length}/score_new"
subfolders = [f.name for f in os.scandir(single_result_folder) if f.is_dir()]
array1 = [0] * length
array2 = [0] * 2
# order = [0] * 7
count = 0
for f in os.scandir(single_result_folder):
    count += 1
    # if count > 50:
    #     break
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
# print(order)
import math

n = 20  # 要求的对数数量
base = 1 / 0.75    # 对数的底

log_values = []  # 存储对数值的列表

log_result = math.log(n, base)

print(log_result)
import math

desired_probability = 0.9  # 期望的事件发生概率
failure_probability = 0.43   # 事件不发生的概率

n = math.ceil(math.log(1 - desired_probability) / math.log(failure_probability))
print(n)