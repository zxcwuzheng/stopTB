import pickle
import pandas as pd


# The input is DataFrame-like object: rows are isolates, columns are gDST and pDST:
# column1: WHO catalog binary predictions (0 represent drug susceptibility, 1 is drug resistance)
# column2: TB profiler binary predictions (0 represent drug susceptibility, 1 is drug resistance)
# column3: SAM-TB binary predictions (0 represent drug susceptibility, 1 is drug resistance)
# column4: GenTB numerical predictions (0~1, the probability of drug resistance)
# column5: MD-CNN numerical predictions (0~1, the probability of drug resistance)
# column6: pDST (0 represent drug susceptibility, 1 is drug resistance)
def prepare_input(gDST_file, pDST_file, drug):

    # read five baseline gDST
    with open(gDST_file, 'rb') as f:
        ALL = pickle.load(f)
        ALL = ALL[drug]
        ALL = pd.DataFrame(ALL)
        df = ALL[[0,1,2,5,6]]
        df.columns = ['WHO_catalog','TBprofiler','SAMTB','GenTB','MDCNN']
        df.index = ALL[7]

    # read pDST
    with open(pDST_file, 'rb') as r:
        ALL_lables = pickle.load(r)
        ALL_lables = ALL_lables[drug]

    # combined 
    df['pDST'] = ALL_lables

    return(df)


#            WHO_catalog  TBprofiler  SAMTB  GenTB     MDCNN  pDST
                                                               
# ERR067576            1           1      1  1.000  0.957823     1
# ERR067577            1           1      1  1.000  0.970510     1
# ERR067578            1           1      1  1.000  0.957917     1
# ERR067580            0           0      0  0.000  0.140778     0
# ERR067581            1           1      1  1.000  0.980599     1
# ...                ...         ...    ...    ...       ...   ...
# 3366                 1           1      1  1.000  0.989694     1
# 4026                 1           1      1  1.000  0.982552     1
# 4557                 1           1      1  1.000  0.995404     1
# 3855                 1           1      1  0.974  0.991523     1
# 3787                 1           1      1  1.000  0.984336     1