"""Model comparison using train/validation/test splits; test remains untouched for selection."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from .features import build_features

XGB_GRID = {'n_estimators':[50,100,300], 'learning_rate':[.01,.1], 'subsample':[.5,.8],
            'colsample_bytree':[.5,1.], 'scale_pos_weight':[1,5,10]}

def auc(y, scores):
    return float(roc_auc_score(y,scores))

def prepare(data, cutoff):
    features=build_features(data,cutoff)
    y=features.pop('target').astype(int)
    features=features.drop(columns=['customer_ID'])
    # Keep consistent numeric columns, with missing values handled by each model.
    features=features.select_dtypes(include='number')
    if not set(y.unique()).issubset({0,1}): raise ValueError('Target must be binary')
    return features,y

def run_xgb(X_train,y_train,X_val,y_val,output,limit=None):
    # Feature selection performed on training set only; validation used for hyperparameters.
    selector=XGBClassifier(n_estimators=300,learning_rate=.5,max_depth=4,subsample=.5,
        colsample_bytree=.5,scale_pos_weight=5,random_state=42,tree_method='hist',n_jobs=-1)
    selector.fit(X_train,y_train)
    selected=X_train.columns[selector.feature_importances_>.005].tolist()
    if not selected: selected=[X_train.columns[int(np.argmax(selector.feature_importances_))]]
    records=[];best=None;best_auc=-1
    for i,params in enumerate(ParameterGrid(XGB_GRID)):
        if limit is not None and i>=limit: break
        m=XGBClassifier(**params,random_state=42,tree_method='hist',n_jobs=-1)
        m.fit(X_train[selected],y_train)
        score=auc(y_val,m.predict_proba(X_val[selected])[:,1])
        records.append({**params,'validation_auc':score})
        if score>best_auc: best_auc=score;best=params
    pd.DataFrame(records).to_csv(output/'xgb_validation_grid.csv',index=False)
    return best,selected,best_auc

def run_nn(X_train,y_train,X_val,y_val,output,limit=None):
    # Import TensorFlow only if NN training requested.
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input,Dense,Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.keras.utils.set_random_seed(42)
    imp=SimpleImputer(strategy='constant',fill_value=0)
    a=imp.fit_transform(X_train).astype('float32')
    b=imp.transform(X_val).astype('float32')
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    a=scaler.fit_transform(a).astype('float32');b=scaler.transform(b).astype('float32')
    rows=[];best=-1;best_config=None
    configs=[(h,n,act,drop,bs) for h in (2,4) for n in (4,6) for act in ('relu','tanh') for drop in (.5,0.) for bs in (100,10000)]
    for i,(layers,nodes,activation,dropout,batch) in enumerate(configs):
        if limit is not None and i>=limit:break
        tf.keras.backend.clear_session()
        model=Sequential([Input(shape=(a.shape[1],))])
        for _ in range(layers):
            model.add(Dense(nodes,activation=activation))
            if dropout: model.add(Dropout(dropout))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='binary_crossentropy')
        model.fit(a,y_train,validation_data=(b,y_val),epochs=20,batch_size=batch,verbose=0,
                  callbacks=[EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)])
        score=auc(y_val,model.predict(b,verbose=0).ravel())
        config=dict(hidden_layers=layers,nodes=nodes,activation=activation,dropout=dropout,batch_size=batch)
        rows.append({**config,'validation_auc':score})
        if score>best:best=score;best_config=config
    pd.DataFrame(rows).to_csv(output/'nn_validation_grid.csv',index=False)
    return best_config,best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True,help='CSV containing customer_ID, S_2, target and Amex features')
    ap.add_argument('--output',default='outputs')
    ap.add_argument('--cutoff',default='2018-03-31')
    ap.add_argument('--run-nn',action='store_true')
    ap.add_argument('--quick',action='store_true',help='Run only 2 configurations per model (smoke test)')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    X,y=prepare(pd.read_csv(args.data),args.cutoff)
    if y.value_counts().min()<3:raise ValueError('Need at least three samples of each class for stratified splitting')
    X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=.15,stratify=y,random_state=42)
    X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,test_size=.15/.85,stratify=y_dev,random_state=42)
    params,selected,val_auc=run_xgb(X_train,y_train,X_val,y_val,out,2 if args.quick else None)
    # Fit final model using train + validation, evaluate only once on held-out test.
    final=XGBClassifier(**params,random_state=42,tree_method='hist',n_jobs=-1)
    final.fit(X_dev[selected],y_dev)
    test_auc=auc(y_test,final.predict_proba(X_test[selected])[:,1])
    joblib.dump({'model':final,'features':selected,'cutoff':args.cutoff},out/'xgb_model.joblib')
    metrics={'xgb_validation_auc':val_auc,'xgb_test_auc':test_auc,'selected_feature_count':len(selected),
             'train_count':len(X_train),'validation_count':len(X_val),'test_count':len(X_test),
             'note':'New evaluation protocol. Not directly comparable to historical report.'}
    if args.run_nn:
        cfg,score=run_nn(X_train,y_train,X_val,y_val,out,2 if args.quick else None)
        metrics['nn_best_validation_auc']=score;metrics['nn_best_config']=cfg
        metrics['nn_note']='NN validation comparison only; final NN refit and test evaluation not implemented.'
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
