import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import xgboost as xgb

X = pd.read_csv("Training_dataset_Original.csv")



with_pca = 1
submit = 1







####Converting string data to float
X = X.replace({'na':np.nan, 'N/A':np.nan, 'missing':np.nan})
X = pd.get_dummies(X, columns = ["mvar47"])
X = X.drop(['application_key'], axis=1)


X[[col for col in X.columns if X[col].dtype != float]] = X[[col for col in X.columns if X[col].dtype != float]].apply(pd.to_numeric)
#print(X.dtypes)
y = X.default_ind

X = X.drop(['default_ind'], axis=1)




#########################splitting to train and test######################
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


##########################Data Cleaning###################################

#myimputer = SimpleImputer(missing_values = "missing", strategy = "median") 
myimputer = Imputer(missing_values = np.nan, strategy = "median")
imp_train_X = myimputer.fit_transform(train_X)
imp_val_X = myimputer.transform(val_X)

imp_train_X = pd.DataFrame(imp_train_X, columns = X.columns)
imp_val_X = pd.DataFrame(imp_val_X, columns = X.columns)
##############################Outiler removal###############################
#for col in imp_train_X.columns:
#
#    Q1 = imp_train_X[col].quantile(0.25)
#    Q3 = imp_train_X[col].quantile(0.75)
#    IQR = Q3 - Q1
#    print("length")
#    print(len(imp_train_X.loc[(imp_train_X[col] > (Q1-1.5*IQR)) & (imp_train_X[col] < (Q3+1.5*IQR))]))
##    if IQR >1000  & len(imp_train_X.loc[(imp_train_X[col] > (Q1-1.5*IQR)) & (imp_train_X[col] < (Q3+1.5*IQR))])>100:
### Filtering Values between Q1-1.5IQR and Q3+1.5IQR
##       
##        imp_train_X = imp_train_X.loc[(imp_train_X[col] > (Q1-1.5*IQR)) & (imp_train_X[col] < (Q3+1.5*IQR))]
##        print(col)
##        print(imp_train_X.shape)
##        print(Q1)
##        print(Q3)
##        print(IQR)

#print(X.corr())




#import seaborn as sns
#corr = X.corr()
#sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
#corr_matrix = df.corr().abs()
#
## Select upper triangle of correlation matrix
#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#
## Find index of feature columns with correlation greater than 0.95
#to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
#
## Drop features 
#df.drop(df.columns[to_drop], axis=1)

#print("Starting minmax scaler")
#scaler = MinMaxScaler()
#imp_train_X = scaler.fit_transform(imp_train_X)
#imp_val_X= scaler.transform(imp_val_X)


###############################FeatureSelection#########################
corr = imp_train_X.corr()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any((upper[column]).abs() > 0.60)]
to_drop = to_drop +['mvar4', 'mvar5']


print(corr)





#############################Data_Exploration##############################
def Data_exp(df, to_drop):
    
    
    df = df.drop(to_drop, axis=1)
    df['diff_in_amt'] = (df.mvar8-df.mvar6)
    df.drop(['mvar6','mvar8'], axis = 1)
#    df['Total_avg_worth'] = (df.mvar14 + df.mvar15)/2
#    df.drop(['mvar14','mvar15'], axis = 1)
    return df
    
    




imp_train_X = Data_exp(imp_train_X, to_drop)
imp_val_X = Data_exp(imp_val_X, to_drop)
#PCA
pca = PCA(n_components=len(imp_train_X.columns))
#pca = PCA(0.95)
pca.fit(imp_train_X)
bullseye=pca.explained_variance_ratio_
print("sum=",np.sum(bullseye))
 
XX_train = pca.fit_transform(imp_train_X)
XX_test= pca.transform(imp_val_X) 


if with_pca ==1 :
    ####################withpca#########################
    classifier = xgb.sklearn.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
           gamma=0, learning_rate=0.05, max_delta_step=0, max_depth=3,
           min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
           objective='binary:logistic', reg_alpha=0, reg_lambda=1,
           scale_pos_weight=1, seed=42, silent=True, subsample=1)
    classifier.fit(XX_train, train_y, early_stopping_rounds=5, 
                   eval_set=[(XX_test, val_y)], verbose=False)
    predictions = classifier.predict(XX_test)
    cfm=confusion_matrix(val_y, predictions)
    print(cfm)
    print((cfm[0,0]+cfm[1,1])/200)
elif with_pca ==0:
    #############################withoutpca###########
    classifier = xgb.sklearn.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
           gamma=0, learning_rate=0.05, max_delta_step=0, max_depth=3,
           min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
           objective='binary:logistic', reg_alpha=0, reg_lambda=1,
           scale_pos_weight=1, seed=42, silent=True, subsample=1)
    
    classifier.fit(imp_train_X, train_y, early_stopping_rounds=5, 
                   eval_set=[(imp_val_X, val_y)], verbose=False)
    
    predictions = classifier.predict(imp_val_X)
    cfm=confusion_matrix(val_y, predictions)
    print(cfm)
    print((cfm[0,0]+cfm[1,1])/200)


###########################finalTraining#########################
if submit == 1:
    final_train = pd.concat([imp_train_X, imp_val_X]) 
    
    final_train_y = pd.concat([train_y,val_y])
    
    pca = PCA(n_components=len(final_train.columns))
    #pca = PCA(0.95)
    pca.fit(final_train)
    bullseye=pca.explained_variance_ratio_
    print("sum=",np.sum(bullseye))
     
    XX_final_train = pca.fit_transform(final_train)
 

    if with_pca ==1 :
        ####################withpca#########################
        classifier1 = xgb.sklearn.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
               gamma=0, learning_rate=0.05, max_delta_step=0, max_depth=3,
               min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
               objective='binary:logistic', reg_alpha=0, reg_lambda=1,
               scale_pos_weight=1, seed=42, silent=True, subsample=1)
        
        classifier1.fit(XX_final_train, final_train_y)
        
        



    elif with_pca ==0:
        #############################withoutpca###########
        classifier1 = xgb.sklearn.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
               gamma=0, learning_rate=0.05, max_delta_step=0, max_depth=3,
               min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
               objective='binary:logistic', reg_alpha=0, reg_lambda=1,
               scale_pos_weight=1, seed=42, silent=True, subsample=1)
        
        classifier1.fit(final_train, final_train_y)
        





Lb_X = pd.read_csv("Leaderboard_dataset.csv")


Lb_X = Lb_X.replace({'na':np.nan, 'N/A':np.nan, 'missing':np.nan})
Lb_X = pd.get_dummies(Lb_X, columns = ["mvar47"])
Lb_app = Lb_X.application_key
Lb_X = Lb_X.drop(['application_key'], axis=1)


Lb_X[[col for col in Lb_X.columns if Lb_X[col].dtype != float]] = Lb_X[[col for col in Lb_X.columns if Lb_X[col].dtype != float]].apply(pd.to_numeric)

imp_lbX = myimputer.transform(Lb_X)

imp_lbX = pd.DataFrame(imp_lbX, columns = Lb_X.columns)

imp_lbX = Data_exp(imp_lbX, to_drop)

if with_pca ==1 :
        ####################withpca##################
        imp_lbX_pca= pca.transform(imp_lbX) 
        pred = classifier1.predict(imp_lbX_pca)
        pred_prob = classifier1.predict_proba(imp_lbX_pca)
elif with_pca == 0 :
        #################without pca#############
        pred = classifier1.predict(imp_lbX)
        pred_prob = classifier1.predict_proba(imp_lbX)
        print(pred_prob)



res = pd.DataFrame(Lb_app)
res['default_ind'] = pred
res['prob'] = pred_prob[:,0]
res = res.sort_values(by = ['prob'], ascending = False)

res.to_csv("AsqR_IITGuwahati_1.csv", header = None, index = None, columns = ['application_key','default_ind'])