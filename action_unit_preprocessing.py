import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

dirs = "dataset/AU/Depresi"

sc = StandardScaler()
result = {}
mean_score = []
std_score =  []
var_score = []


for f in os.listdir("dataset/AU/Depresi"):
    if f.endswith('.csv'):
        fname = os.path.join("dataset/AU/Depresi",f)
        # print()

        data = pd.read_csv(fname,skipinitialspace=True)
        data = sc.fit_transform(data)
        df = pd.DataFrame(data, columns=["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU25_r","AU26_r","AU04_c","AU12_c","AU15_c","AU23_c","AU28_c","AU45_c"])

        # for col in df.columns:
        mn = np.mean(df['AU01_r'])
        vr = np.var(df['AU01_r'])
        st = np.std(df['AU01_r'])

        mean_score.append(mn)
        std_score.append(st)
        var_score.append(vr)


class1 = np.zeros(len(mean_score))
class1.fill(1)
dataset_depresi = np.column_stack((mean_score, std_score, var_score, class1))

mean_score = []
std_score =  []
var_score = []



for f in os.listdir("dataset/AU/NonDepresi"):
    if f.endswith('.csv'):
        fname = os.path.join("dataset/AU/NonDepresi",f)
        # print()

        data = pd.read_csv(fname,skipinitialspace=True)
        data = sc.fit_transform(data)
        df = pd.DataFrame(data, columns=["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU25_r","AU26_r","AU04_c","AU12_c","AU15_c","AU23_c","AU28_c","AU45_c"])

        # for col in df.columns:
        mn = np.mean(df['AU01_r'])
        vr = np.var(df['AU01_r'])
        st = np.std(df['AU01_r'])

        mean_score.append(mn)
        std_score.append(st)
        var_score.append(vr)

class0 = np.zeros(len(mean_score))
dataset_non_depresi = np.column_stack((mean_score, std_score, var_score, class0))
# dataset_non_depresi['Class'] = 0
dataset = np.row_stack((dataset_depresi, dataset_non_depresi))


# print(class1)
print(dataset)


