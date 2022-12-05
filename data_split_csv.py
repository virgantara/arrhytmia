import numpy as np
import pandas as pd
import glob
import os

parent_dir = 'dataset/YaseenDeepLearning'
dirs = os.listdir(parent_dir)

for dir in dirs:
    # print()
    csv_files = []
    list_class = os.listdir(os.path.join(parent_dir,dir))
    for cls in list_class:
        print(dir,cls)
        for fn in glob.glob(os.path.join(parent_dir,dir,cls,"*.wav")):
            print(fn)
            csv_files.append(np.array([
                fn, dir, cls
            ]))

    df = pd.DataFrame(csv_files)
    df.to_csv('yaseen_dl_'+ dir + '.csv', index=False, header=False)
