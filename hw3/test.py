import numpy as np

a=np.array([[1, 2, 3],
       [5, 5, 5],
       [5, 5, 5],
       [5, 5, 5]])
print(a[0,0])
print(np.all(a==a[0,0]) )#所有的元素都一样吗？

print(np.amin(a[:,1])  )#找最小的元素

import treePlotter

myTree = treePlotter.retrieveTree(0)
treePlotter.createPlot(myTree)

plt.figure(frameon=False)



