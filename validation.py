
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from ReadInput import prepare_input
from sn_sp import sn_sp


#------
# mian
#------
# first-line drugs
all_drugs = ['RIFAMPICIN', 'ISONIAZID',
             'PYRAZINAMIDE', 'ETHAMBUTOL']

for drug in all_drugs:

    df = prepare_input('./input_data/external_dataset.pkl',
                       './input_data/external_labels.pkl',
                       drug)
    X = df.iloc[:,:5]
    y = df['pDST']
    

    # load trained model 
    model_path = f'trained_model/{drug}_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # load best cutoff 
    model_cutoff = f'trained_model/{drug}_cutoff.pkl'
    with open(model_cutoff, 'rb') as f:
        cutoff = pickle.load(f)

    # prediction in external validation set
    pred = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, pred)
    sens, spec = sn_sp(y, pred, cutoff)[:2]

    # store result
    valid_info = pd.DataFrame({'strain':y.index,
                                'pDST':y,
                                'WHO catalog': X['WHO_catalog'],
                                'TBprofiler': X['TBprofiler'],
                                'SAM-TB':X['SAMTB'],
                                'GenTB':X['GenTB'],
                                'MDCNN':X['MDCNN'],
                                'stacking':sn_sp(y, pred, cutoff)[3]})
    metric = [auc, sens, spec]

    saved_result = {'Valid set': valid_info,
                    'Valid metric': pd.DataFrame(metric,index=['AUC','Sens','Spec'])}
    
    with pd.ExcelWriter(f'./external_validation/{drug}_output.xlsx') as writer:
        for sheet_name, df in saved_result.items():
            df.to_excel(writer, sheet_name=sheet_name)