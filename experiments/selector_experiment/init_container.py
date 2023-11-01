import os 
import pickle
import random 
import time

class Container:
    def __init__(self, cur_subfolders):
        self.cur_subfolders = cur_subfolders
        self.new_subfolders = []
        self.start_idx = 0

if __name__ == '__main__':
    target_folder = "/scorpio/home/shenao/myProject/Canvas/experiments/collections/preliminary_kernels_selected"  
    if not os.path.isdir(target_folder):
        print('Given path must be a directory')
    FILE = f'{target_folder}/my_container.pkl'
    random.seed(time.time())
    cur_subfolders = [f.name for f in os.scandir(target_folder) if f.is_dir()]
    random.shuffle(cur_subfolders)
    my_container = Container(cur_subfolders)
    print('len(my_container.cur_subfolders): ', len(my_container.cur_subfolders))
    file = open(FILE, 'wb')
    pickle.dump(my_container, file)
    file.close()
    