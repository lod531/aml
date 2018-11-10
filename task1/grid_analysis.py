from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle

grid_search = pickle.load(open('small_grid_search.tmp', 'rb'))
grid_search_results = grid_search.cv_results_
print(type(grid_search_results))
frame = pd.DataFrame.from_dict(grid_search_results)
print(list(frame))
print(frame.shape)
print("YEPPPP")
