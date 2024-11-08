# Advantages of updated WHO mutation catalogue combined with existing whole-genome sequencing-based methods in diagnosis of drug-resistant tuberculosis

An ensemble model leveraging a stacking approach, integrated with the updated WHO catalog,TB profiler(v5.0.0), SAM-TB(v1), GenTB-RF(v1), and MD-CNN(v1), has been developed for the prediction of antibiotic resistance in Mycobacterium tuberculosis genomes.

## requirments
The model is trained using Python (v3.9.18) and scikit-learn package (v1.3.2)


## running
### Input data
Prior to executing our model, it is essential to have acquired the prediction from the five baseline methodologies for each individual isolate. Please organize these results in the sequence specified as follows:

- column1: WHO catalog binary predictions (0 represent drug susceptibility, 1 is drug resistance)
- column2: TB profiler binary predictions (0 represent drug susceptibility, 1 is drug resistance)
- column3: SAM-TB binary predictions (0 represent drug susceptibility, 1 is drug resistance)
- column4: GenTB numerical predictions (0~1, the probability of drug resistance)
- column5: MD-CNN numerical predictions (0~1, the probability of drug resistance)
- column6: pDST (0 represent drug susceptibility, 1 is drug resistance)

For example:
```
            WHO_catalog  TBprofiler  SAMTB  GenTB     MDCNN  pDST
                                                               
ERR067576            1           1      1  1.000  0.957823     1
ERR067577            1           1      1  1.000  0.970510     1
ERR067578            1           1      1  1.000  0.957917     1
ERR067580            0           0      0  0.000  0.140778     0
ERR067581            1           1      1  1.000  0.980599     1
```

### Executing
All the trained model in the `trained_model` subdirectories. 
1. load the trained model using `pickle` module.
```
with open(model_path, 'rb') as f:
    model = pickle.load(f)
```
2. input data and prediction
```
X = pd.read_csv(input.csv)
y_pred = model.predict_proba(X)[:,1]
```
3. get binary predictions based on best threshold
```
model_cutoff = 'trained_model/RIFAMPICIN_cutoff.pkl'
    with open(model_cutoff, 'rb') as f:
        cutoff = pickle.load(f)
y = (pred > cutoff).astype('int')
```

## List of files and directories
`ReadInput.py`: process input data for stacking ensembl model

`sn_sn.py`: calculate sensitivity, specificity and best threshold

`trained.py`: to train model in training set and evaluate in test set

`validation.py`: external validation for trained model

`input_daat`: raw input data

`result`: output result files created by running the models on train and test set

`trained_model`: contain trained model for 10 drugs

`external_validation`: output result files created by running the models on external set

`SuppTable`: store other supplementary table
