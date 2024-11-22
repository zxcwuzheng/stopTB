from ReadInput import prepare_input
from sn_sp import sn_sp
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn import tree
from sklearn.metrics import roc_auc_score
import pickle
import pandas as pd


def training(X, y, drug):
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        stratify=y,
                                                        test_size=0.25,
                                                        random_state=67)
    
    # Initialize a decision tree classifier
    clf = tree.DecisionTreeClassifier(class_weight='balanced',
                                      random_state=0)
    

    # Set the hyper-parameters for cross-validation
    param_grid = {
        'splitter': ['best','random'],
        'criterion': ['gini', 'entropy'],
        'max_features':[None, 'log2','sqrt'],
        'max_depth': [None,2,3,4,5]
    }

    # creat GridSearchCV object
    grid_search = GridSearchCV(clf, 
                               param_grid,
                               cv=StratifiedKFold(n_splits=10,
                                                  random_state=0,
                                                  shuffle=True),
                               n_jobs=2,
                               scoring='roc_auc')
    
    # start hyper-parameter search
    grid_search.fit(X_train, y_train)


    # Re-train the model on the entire training set using the best hyper-parameters
    final_clf = tree.DecisionTreeClassifier(**grid_search.best_params_, 
                                            random_state=0)
    final_clf.fit(X_train, 
                  y_train)
    
    # save tranied model
    with open(f'./trained_model/{drug}_model.pkl', 'wb') as f:
        pickle.dump(final_clf, f)

    # obtain optiomal threshold accoreding max Youden index
    train_pred = final_clf.predict_proba(X_train)[:,1]
    best_cutoff = sn_sp(y_train,
                        train_pred,
                        'best')[2]
    
    # save best cutoff
    with open(f'./trained_model/{drug}_cutoff.pkl', 'wb') as f:
        pickle.dump(best_cutoff, f)


    # use trained model whith best cutoff to predict test set
    y_pred = final_clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred)
    sens, spec = sn_sp(y_test, y_pred, best_cutoff)[:2]

    # output for test set
    test_info = pd.DataFrame({'strain':y_test.index,
                                'pDST':y_test,
                                'WHO catalog': X_test['WHO_catalog'],
                                'TBprofiler': X_test['TBprofiler'],
                                'SAM-TB':X_test['SAMTB'],
                                'GenTB':X_test['GenTB'],
                                'MDCNN':X_test['MDCNN'],
                                'stacking':sn_sp(y_test, y_pred, best_cutoff)[3]})

    # output for training set
    train_info = pd.DataFrame({'strain':y_train.index,
                                'pDST':y_train,
                                'WHO catalog': X_train['WHO_catalog'],
                                'TBprofiler': X_train['TBprofiler'],
                                'SAM-TB':X_train['SAMTB'],
                                'GenTB':X_train['GenTB'],
                                'MDCNN':X_train['MDCNN'],
                                'stacking':sn_sp(y_train, train_pred, best_cutoff)[3]})


    # output auc, sens, spec of test set
    metric = [auc, sens, spec]
    return train_info, test_info, metric



#------
# mian
#------
# Ten drugs
all_drugs = ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE', 'ETHAMBUTOL', 
             'STREPTOMYCIN', 'LEVOFLOXACIN', 'CAPREOMYCIN', 'AMIKACIN', 
             'KANAMYCIN', 'ETHIONAMIDE']

for drug in all_drugs:
    
    df = prepare_input('./input_data/WHO_dataset_gDST.pkl',
                       './input_data/WHO_dataset_pDST.pkl',
                       drug)
    X = df.iloc[:,:5]
    y = df['pDST']

    # perform training and predict in test set
    train_info, test_info, metric = training(X, y, drug)
    saved_result = {'Train set': train_info,
                    'Test set': test_info,
                    'Test metric': pd.DataFrame(metric,index=['AUC','Sens','Spec'])}

    # save related output
    with pd.ExcelWriter(f'result/{drug}_output.xlsx') as writer:
        for sheet_name, df in saved_result.items():
            df.to_excel(writer, sheet_name=sheet_name)