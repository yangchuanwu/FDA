import os
import numpy as np

_data_path = '../Data'
img_path = os.path.join(_data_path, 'SE')
val_path = os.path.join(_data_path, 'val.txt')
test_path = os.path.join(_data_path, 'test.txt')

files = os.listdir(img_path)

get_key = lambda i: int(i.split('.')[0])
filename = sorted(files, key=get_key)

np.random.shuffle(filename)
test_size = int(0.2 * len(files))
val_size = int(0.2 * len(files))
train_size = 6

test = filename
val = filename[train_size:val_size+train_size]

for j in range(train_size):
    train_path = os.path.join(_data_path, 'train' + str(j+1) + '.txt')
    with open(train_path, 'w') as f:
        for i in filename[:j+1]:
            f.write(i + '\n')

with open(val_path, 'w') as f2, open(test_path, 'w') as f3:
    for j in val:
        f2.write(j + '\n')
    for k in test:
        f3.write(k + '\n')

print('success!')

