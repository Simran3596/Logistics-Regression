import  pandas as pd
from sklearn.metrics import  *
import os


data=pd.read_csv("C:\\ML\\loan prediction.csv")

#preprocessing
data.isnull().sum()

#Missing categorical values with mode
data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
data['Married'].fillna(data['Married'].mode()[0],inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace=True)

#Median ,not mean because there are outliers in loan amount
data['LoanAmount'].fillna(data['LoanAmount'].median(),inplace=True)

#Dummy variable
data1= data.iloc[:,1:-1]
data1=pd.get_dummies(data1,columns=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area'],drop_first=True)

X=data1.values
y=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=10)

#Model Building & Prediction
from sklearn.linear_model import LogisticRegression
log_reg= LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred= log_reg.predict(X_test)

#y_pred= log_reg.predict_proba(X_test)

#Model Evaluation
print(" The accuracy of model on training dataset is {:.4f} ".format(log_reg.score(X_train,y_train)))
print(" The accuracy of model on testing dataset is {:.4f} ".format(log_reg.score(X_test,y_test)))

#confusion matrix
from sklearn.metrics import confusion_matrix
#cm= confusion_matrix(y_test,y_pred)
cm= confusion_matrix(y_test,y_pred,labels=['Y','N'])

print(cm)


#df1= pd.DataFrame(y_test,y_pred)
#df1.to_csv("TestandPred.csv")

#cross validation
log_reg1=LogisticRegression(max_iter=50)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(log_reg1,X,y,cv=10)
print ('{:.3f}'.format(accuracies.mean()))



