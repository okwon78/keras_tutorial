import numpy as np
from sklearn.preprocessing import MinMaxScaler

my_list = np.array(range(12))

print(my_list.shape)
print(my_list)
my_list = my_list.reshape(-1, 1)
print(my_list.shape)
print(my_list)
