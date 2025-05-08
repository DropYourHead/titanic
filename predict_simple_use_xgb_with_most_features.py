import xgboost as xgb
import pandas as pd
import numpy as np
print(xgb.__version__)
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
genderSubmission = pd.read_csv('gender_submission.csv')
model = xgb.XGBClassifier(
    n_estimators=100,  
    max_depth=9,       
    learning_rate=0.1, 
    objective='binary:logistic',  
    random_state=42
)
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S' : 2}).fillna(-1)
test['Embarked'] = test['Embarked'].map({'C': 0, 'Q': 1, 'S' : 2}).fillna(-1)
train['Cabin'] = train['Cabin'].fillna("A0")
test['Cabin'] = test['Cabin'].fillna("A0")
train_num = (
    train["Cabin"]
    .str.extract(r'([A-Za-z])(\d+)')[1]  
    .fillna(0)  
    .astype(int)                         
)
test_num = (
    test["Cabin"]
    .str.extract(r'([A-Za-z])(\d+)')[1]  
    .fillna(0)  
    .astype(int)                         
)
max_train_num = max(train_num)
max_test_num = max(test_num)
train_letter = train['Cabin'].str[0].map(ord) - ord('A')
test_letter = test['Cabin'].str[0].map(ord) - ord('A')
train['Cabin'] = train_letter * max_train_num + train_num
test['Cabin'] = test_letter * max_test_num + test_num
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Ticket']
train['Ticket'] = np.where(
    train['Ticket'].str[0].str.isalpha(),  
    0,                                     
    1                                      
)
test['Ticket'] = np.where(
    test['Ticket'].str[0].str.isalpha(),  
    0,                                     
    1                                      
)
exit()
X_train = train[features]
y_train = train['Survived']
X_test = test[features]
model.fit(X_train, y_train)
X_val = test[features]

# predict
y_pred = model.predict(X_val)

#generate outcome
submission = pd.DataFrame({
    "PassengerId": genderSubmission["PassengerId"],  # 从测试数据读取 PassengerId
    "Survived": y_pred                    # 预测结果
})

submission.to_csv("submission.csv", index=False)



