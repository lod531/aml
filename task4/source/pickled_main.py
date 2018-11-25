from get_data import get_videos_from_folder,get_target_from_csv
import os
import numpy as np
from utils import save_solution
import pickle



dir_path = os.path.dirname(os.path.realpath(__file__))
train_folder = os.path.join(dir_path,"../train/")
test_folder = os.path.join(dir_path,"../test/")

x_train = pickle.load(open('x_train.pickle', 'rb'))
y_train = pickle.load(open('y_train.pickle', 'rb'))
x_test = pickle.load(open('x_test.pickle', 'rb'))
print(len(x_train[0]))



my_solution_file = os.path.join(dir_path,'../solution.csv')
dummy_solution = 0.1*np.ones(len(x_test))
save_solution(my_solution_file,dummy_solution)
