import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

t1 = scipy.sparse.load_npz("../data/rnac/contact_map.txt.npz")
# t2 = scipy.sparse.load_npz("../data/rnac/contact_map_100.txt.npz")
predict_input = np.load("../data/rnac/pred_list.npy")
result = np.load("../data/rnac/result.txt.npy").reshape([-1, 5, 41])[:, [1, 2, 3, 0, 4], :]


def draw(data):
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()

for index, i in enumerate(t1):
    k2 = np.array(i.toarray())
    # k1 = np.array(t2[index, :].toarray())
    pred = np.reshape(predict_input[index, :], [-1, 1, 1])

    # l1 = np.reshape(k1, [100, 5, 41])
    l2 = np.reshape(k2, [10, 5, 41])

    # print(pred)

    l2 = l2 * pred

    # l1 = np.mean(l1, axis=0)
    l2 = np.sum(l2, axis=0)
    #
    # print(l1)
    # print(l2)
    l1 = result[index, :]
    draw(l1)
    draw(l2)
    value = np.sum((l1 - l2) * (l1 - l2))
    print(1. * value / 5 / 41)
