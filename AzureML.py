# Creating a Notebook in Azure Machine Learning
## Used the following documentation https://learn.microsoft.com/en-us/training/modules/analyze-climate-data-with-azure-notebooks/
## Created a Scatter plot in Azure Notebooks using the Python - Azure ML module.

Code: import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()
yearsBase, meanBase = np.loadtxt('graph (1).csv', delimiter=',', usecols=(0, 1), unpack=True, skiprows=1)
years, mean = np.loadtxt('graph (2).csv', delimiter=',', usecols=(0, 1), unpack=True, skiprows=1)
plt.scatter(yearsBase, meanBase)
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)
plt.show()


![image](https://github.com/user-attachments/assets/16d1ab47-e674-4f3e-9ac8-66a7bb6337a2)
