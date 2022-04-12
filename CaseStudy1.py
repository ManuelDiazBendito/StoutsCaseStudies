import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import copy
import math

Seed = 1

#Load the Advertising dataset and save radio as
#the features and sales as the labels:
    
df = pd.read_csv("loans_full_schema.csv")

Data = df.to_numpy()

Column_Names = df.columns
NonNumericCols = []
        
#interest_rate prediciton and data visualization:

Y = Data[:,Column_Names=="interest_rate"]
for col in range(len(Data[0,:])):
    try:
        plt.scatter(Y,Data[:,col], color = 'blue', marker = "s" )
        plt.title("Interest rate vs " + Column_Names[col])
        plt.xlabel("Interest_Rate")
        plt.ylabel(Column_Names[col])
        plt.show()
    except:
        print("No numeric data to show in ", Column_Names[col])
        
### KNN training process: ###
    
#Dataset preparation:
X_letters = Data[:,Column_Names=="grade"]
X = copy.copy(X_letters)

#grade feature engineering process:
for i in range(len(X_letters)):
    if X_letters[i] == 'A':
        X[i] = 1
    elif X_letters[i] == 'B':
        X[i] = 2
    elif X_letters[i] == 'C':
        X[i] = 3
    elif X_letters[i] == 'D':
        X[i] = 4
    elif X_letters[i] == 'E':
        X[i] = 5
    elif X_letters[i] == 'F':
        X[i] = 6
    elif X_letters[i] == 'G':
        X[i] = 7

#Data partitioning process:
X_train, X_test,  y_train, y_test= train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=Seed)
X_test, X_validate,  y_test, y_validate= train_test_split(X_test, y_test, train_size=0.5, test_size=0.5, random_state=Seed)

#Declare important variables
BestScore=0
AllScores=[]

#kNN model initialization:
kNN=KNeighborsRegressor()

#HYPERPARAMETER TUNING PROCESS:
for k in range(1,100):
    
    #Train model with training dataset and selected features
    kNN.set_params(n_neighbors=k)
    kNN.fit(X_train,y_train) 
    NewScore=kNN.score(X_validate,y_validate) 
    
    #Get best score for current hyperparameter
    if  BestScore<NewScore:
        BestScore=NewScore
        AllScores.append(NewScore)
        FinalK=k
    else: 
        AllScores.append(NewScore)

#Plot each iteration for finding best hyperparam.
plt.scatter(range(1,100),AllScores,color='blue', marker = "s" )
plt.xlabel("k Values")
plt.ylabel("Score")
plt.title("kNN - Final k value = "+str(FinalK)+" with ValidationScore = "+str(BestScore))
plt.show()

#Test the trained model
kNN.set_params(n_neighbors=FinalK)
kNN.fit(X_train,y_train) 
print("Test set Score for the kNN model is = ", kNN.score(X_test,y_test))

### Random Forest training process: ###

#Dataset preparation
X = Data[:,Column_Names != "interest_rate"]
Y = Data[:,Column_Names == "interest_rate"]

ColsToDel = []
for cols in range(len(X[0,:])):
    for rows in range(len(X[:,0])):
        if type(X[rows,cols]) == str:
            ColsToDel.append(cols)
            break
        
#Delete non numeric features:
X=np.delete(X,ColsToDel,1)
NewColNames=np.delete(Column_Names,ColsToDel,0)

#Reassign nan values:
for i in range(len(X[:,0])):
    for j in range(len(X[0,:])):
        if math.isnan(X[i,j]):
            X[i,j] = -10000
  
#Adjust target vector:
Y = np.ravel(Y)   
 
#Partition daa
X_train, X_test,  y_train, y_test= train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=Seed)

#Cross validation process initialization
nof_folds=10
kf = KFold(n_splits=nof_folds)

#Grid Set of hyperparameters initialization:
nof_estimators = [1, 10, 50, 100, 200]
criterion =  ['mse' , 'mae' , 'poisson']
min_samples_leaf = [1, 2, 5, 10, 20]
min_samples_split = [2]
max_features = ['auto']

nof_iters = 5*3*5*1*1

#Best score variables initialization:
score=0
final_scores=[]
best_score=0
curr_iter=1

#Grid search processs:
for nof_e in nof_estimators:
    for crit in criterion:
        for min_s_l in min_samples_leaf:
            for min_s_s in min_samples_split:
                for max_f in max_features:
                    #MLP initialization:
                    RFR = RandomForestRegressor(random_state = Seed , n_estimators = nof_e, criterion = crit, min_samples_leaf = min_s_l, min_samples_split = min_s_s, max_features = max_f)
                    Train_Scores=[]
                    Val_Scores=[]
                    #Cross-validation procedure:
                    for train_indices in kf.split(X_train):
                        RFR.fit(X_train[train_indices[0]], y_train[train_indices[0]])
                        Train_Scores.append(RFR.score(X_train[train_indices[0]], y_train[train_indices[0]]))
                        Val_Scores.append(RFR.score(X_train[train_indices[1]], y_train[train_indices[1]]))
                        score=score+RFR.score(X_train[train_indices[1]], y_train[train_indices[1]])
                    #Logging information
                    print("Iter number ",curr_iter,"out of ",nof_iters)
                    print("Hyperparameters set average score is: ",score/nof_folds)
                    curr_iter=curr_iter+1
                    #Best validation score calculation:
                    final_scores.append(score/nof_folds)
                    if (score/nof_folds)>best_score:
                        best_score = score/nof_folds
                        best_score_model = RFR
                        FinalTrainScores=Train_Scores
                        FinalValScores=Val_Scores
                    score=0
    
#Final outputs
print("Manually Grid Search implementation best model hyperparameters are:", best_score_model.get_params())
print("Manual Hyperparameter Grid Search implementation final best train scores are:",FinalTrainScores)
print("Manual Hyperparameter Grid Search implementation final best validation scores are:",FinalValScores)
print("Manual Hyperparameter Grid Search implementation final best test score is:",best_score_model.score(X_test,y_test))
