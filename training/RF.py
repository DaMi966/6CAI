import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, cross_validate
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, RocCurveDisplay, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris  # 示例数据集
import joblib

csv_file_path = 'E:\\Paper\\探测实验\\模型实验\\训练数据\\1013训练\\data-16\\data-new.csv'

# 读取数据
data = pd.read_csv(csv_file_path)

# 打印前几行数据
print(data.head())

# 提取特征和标签
X = data[['count','variance','range','max_diff','variance_diff','5th_diff_p',
          'mean_diff_p','count_type_1_or_3','variance_type_1_or_3','range_type_1_or_3',
          'min_diff_type_1_or_3','variance_diff_type_1_or_3','95th_diff_type_1_or_3',
          '5th_diff_type_1_or_3','mean_diff_type_1_or_3','variance_diff_p_type_1_or_3',
          'max_diff_p_type_1_or_3','min_diff_p_type_1_or_3','95th_diff_p_type_1_or_3',
          'mean_diff_p_type_1_or_3','iid_really_macderived_ratio','iid_embeddedipv4_32_ratio',
          'iid_random_ratio','iid_really_macderived_over_total_ratio'
]]
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义随机森林模型
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 使用交叉验证评估模型
cv_results = cross_validate(rf_classifier, X_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

# 打印交叉验证结果
print("Cross-Validation Results:")
print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.3f} ± {np.std(cv_results['test_accuracy']):.3f}")
print(f"Precision: {np.mean(cv_results['test_precision']):.3f} ± {np.std(cv_results['test_precision']):.3f}")
print(f"Recall: {np.mean(cv_results['test_recall']):.3f} ± {np.std(cv_results['test_recall']):.3f}")
print(f"F1-Score: {np.mean(cv_results['test_f1']):.3f} ± {np.std(cv_results['test_f1']):.3f}")
print(f"AUC: {np.mean(cv_results['test_roc_auc']):.3f} ± {np.std(cv_results['test_roc_auc']):.3f}")

# 训练模型
rf_classifier.fit(X_train, y_train)

# 预测并评估模型
y_pred = rf_classifier.predict(X_test)
y_prob = rf_classifier.predict_proba(X_test)[:, 1]  # 获取正类的概率

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = auc(fpr, tpr)

# 打印各项指标
print(f"\nTest Set Evaluation:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"AUC: {auc_score:.3f}")

# 自定义函数格式化分类报告
def format_classification_report(report):
    lines = report.split('\n')
    formatted_lines = []
    for line in lines:
        if 'avg' in line or 'accuracy' in line:
            parts = line.split()
            formatted_parts = [parts[0]] + [f"{float(p):.3f}" if p.replace('.', '', 1).isdigit() else p for p in parts[1:]]
            formatted_lines.append(' '.join(formatted_parts))
        else:
            formatted_lines.append(line)
    return '\n'.join(formatted_lines)

# 打印分类报告
report = classification_report(y_test, y_pred, digits=3)
formatted_report = format_classification_report(report)
print("\nClassification Report:\n", formatted_report)

# 特征重要性
feature_importances = rf_classifier.feature_importances_
features = X.columns

# 按照重要性排序输出特征
sorted_indices = np.argsort(feature_importances)[::-1]
for index in sorted_indices:
    print(f"Feature: {features[index]}, Importance: {feature_importances[index]}")

# 绘制学习曲线
def plot_learning_curve(estimator, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))  # 调整图表大小

    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples", fontsize=35)  # 增加横坐标字体大小
    axes.set_ylabel("Score", fontsize=35)  # 增加纵坐标字体大小
    axes.xaxis.labelpad = 20
    axes.yaxis.labelpad = 20
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Test score")
    axes.legend(loc='lower right', fontsize=35)  # 增加图例字体大小
    axes.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    return plt

plt.figure(figsize=(12, 8))  # 调整图表大小
plot_learning_curve(rf_classifier, X_train, y_train, cv=5)
plt.xticks(fontsize=25)  # 增加x轴刻度字体大小
plt.yticks(fontsize=25)  # 增加y轴刻度字体大小

plt.tight_layout()  # 自动调整子图参数
plt.savefig('./png/RF-new.png', dpi=400)
plt.show()
# 保存模型到文件
joblib.dump(rf_classifier, './pkl/random_forest_model-1017-new.pkl')
print("Model saved.")