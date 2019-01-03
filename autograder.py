import machine_learning_hw1 as mlhw
import numpy as np
import math

mlhw.train("training/training.tfrecords")
pred, gt = mlhw.test("validation/validation.tfrecords")
assert len(pred) == len(gt), "The length of prediction and ground truth should be the same"
correct = np.count_nonzero(np.array(pred) == np.array(gt))
print("Accuracy: {}%".format(float(correct) / len(pred) * 100))
