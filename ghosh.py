import numpy as np
import pandas as pd
headers = ["atr"+str(i) for i in range(1,300)]
headers = np.append(headers,'class')
data = pd.read_csv("ghosh_features_yaseen_data.csv",names=headers)
data = data.fillna(method='ffill')
data.to_csv('ghosh_features_yaseen.csv', index=False,header=False)
# print(data.head())