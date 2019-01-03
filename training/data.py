import numpy as np
import cv2

data = []
label = []
for i in range(1,11):
    if i < 10:
        for j in range(51,116):
            if j < 100:
                img = cv2.imread('./sample00{}/img00{}-000{}.png'.format(i,i,j), 0)
                data.append(img)
                label.append(i-1)
            else:
                img = cv2.imread('./sample00{}/img00{}-00{}.png'.format(i,i,j), 0)
                data.append(img)
                label.append(i-1)
    else:
        for j in range(51,116):
            if j < 100:
                img = cv2.imread('./sample0{}/img00{}-000{}.png'.format(i,i,j), 0)
                data.append(img)
                label.append(i-1)
            else:
                img = cv2.imread('./sample0{}/img00{}-00{}.png'.format(i,i,j), 0)
                data.append(img)
                label.append(i-1)
labels = []
for i in range(len(label)):
    l = np.zeros(10)
    l[label[i]] = 1
    labels.append(l)

c = list(zip(data, labels))
np.random.shuffle(c)
data, labels = zip(*c)
data = np.array(data)
labels = np.array(labels)
np.save('../train_data.npy', data)
np.save('../train_label.npy', labels)
