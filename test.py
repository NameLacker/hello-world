import numpy as np
# import heapq

# a = np.array([[1,np.nan,2.1,5],[1.3,2,3,2],[1,2,6,2],[2, 1,7,2]], dtype='float32')


##### 处理nan，inf######
# nan = np.isnan(a)
# print(nan)
# a[nan] = 0
# print(a)
##### 处理nan，inf######

##### 找最大值索引######
# a = np.array([2,34,1,5,5])
# c = a.argsort()[-3:][::-1]
# c = (-a).argsort()[:2]
# print(c)

# a = np.random.randint(0, 10, 6)
# b = np.random.randint(0, 10, 6)
# c = np.random.randint(0, 10, 6)
# d = np.row_stack((a, b))
# d = np.row_stack((c ,d))

##### 处理nan，inf######



a = np.array([
    [1,2,3],
    [2,3,4],
    [2,3,5]
])
print(a[:,2])