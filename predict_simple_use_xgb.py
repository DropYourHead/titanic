import xgboost as xgb
import pandas as pd
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
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
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



