import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
# 从文件加载模型
# 随机森林
loaded_rf = joblib.load('./pkl/random_forest_model-1017-new.pkl')
# 支持向量机 (Support Vector Machines, SVM)
# loaded_model = joblib.load('svm_model.pkl')
# K-近邻 (K-Nearest Neighbors, KNN)
# loaded_rf = joblib.load('./pkl/knn_new_model-1011.pkl')
# 决策树 (Decision Trees)
# loaded_model = joblib.load('decision_tree_model.pkl.pkl')
# 逻辑回归 (Logistic Regression)
# loaded_model = joblib.load('logistic_regression_model.pkl')
# 朴素贝叶斯 (Naive Bayes)
# loaded_model = joblib.load('naive_bayes_model.pkl')
csv_file_path = 'E:\\Paper\\探测实验\\模型实验\\验证数据\\1018验证\\data-new\\output_Y_Y_29-new.csv'

# 确保使用pd.read_csv()正确读取数据，并检查数据类型
new_data = pd.read_csv(csv_file_path)


# 现在应该可以成功打印前几行数据
print(new_data.head())

# 假设使用svm标准化器
# scaler = joblib.load('svm-scaler.pkl')
# new_data_scaled = scaler.transform(new_data)

# 假设使用knn标准化器
# scaler = joblib.load('./pkl/knn_new_scaler-1011.pkl')
# new_data_scaled = scaler.transform(new_data)

# 假设使用dtree标准化器
# scaler = joblib.load('dtree-scaler.pkl')
# new_data_scaled = scaler.transform(new_data)

# 假设使用LR标准化器
# scaler = joblib.load('logreg-scaler.pkl')
# new_data_scaled = scaler.transform(new_data)

# 假设使用NB标准化器
# scaler = joblib.load('naive-bayes-scaler.pkl')
# new_data_scaled = scaler.transform(new_data)
# 使用加载的模型进行预测
new_predictions = loaded_rf.predict(new_data)
# 经过标准化后的预测
# new_predictions = loaded_rf.predict(new_data_scaled)

# 将预测结果添加到DataFrame作为新的一列'target'
new_data['target'] = new_predictions

# 定义新CSV文件的路径，例如在原文件名后加上'_predicted'
new_csv_file_path = csv_file_path.rsplit('.', 1)[0] + '_predicted.csv'

# 保存带有预测结果的新DataFrame到新的CSV文件
new_data.to_csv(new_csv_file_path, index=False)

print(f"Predictions have been saved to a new CSV file: {new_csv_file_path}")