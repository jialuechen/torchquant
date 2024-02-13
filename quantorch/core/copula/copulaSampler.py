from numpy import random
from random import sample
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import math

class Copula():
    def __init__(self,data):
        self.data = np.array(data)
        if(len(data)<2):
            raise  Exception('input data must have multiple samples')
        if not isinstance(data[0], list):
            raise  Exception('input data must be a 2D array')
        self.cov = np.cov(self.data.T)
        if 0 in self.cov:
            raise  Exception('Data not suitable for Copula. Covarience of two column is 0')
        self.normal = stats.multivariate_normal([0 for i in range(len(data[0]))], self.cov,allow_singular=True)
        self.norm = stats.norm()
        self.var = []
        self.cdfs = []
    def gendata(self,num):
        self.var = random.multivariate_normal([0 for i in range(len(self.cov[0]))], self.cov,num)
        
        for i in self.var:
            for j in range(len(i)):
                i[j]= i[j]/math.sqrt(self.cov[j][j])
        self.cdfs = self.norm.cdf(self.var)
        data = [ [ np.percentile(self.data[:,j],100*i[j]) for j in range(len(i))] for i in self.cdfs ]
        return data

    


