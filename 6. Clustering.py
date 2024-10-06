from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot

X,y=make_classification(n_samples=1000,
                        n_feature=2,n_informative=2,n_redundent=0,
                        n_clusters_per_class=1,random_stats=4)

for class_value in range(2):
    row_ix=where(y==class_value)
    pyplot.scatter(X[row_ix,0],X[row_ix,1])

pyplot.show()