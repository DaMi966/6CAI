import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

csv_file_path = 'E:\\Paper\\探测实验\\模型实验\\训练数据\\1013训练\\data-16\\data-new.csv'

# 读取 CSV 文件
data = pd.read_csv(csv_file_path)
print(data.head())

# 处理缺失值
if data.isnull().values.any():
    print("Data contains NaN values. Handling NaN values...")
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
else:
    data_imputed = data

# 提取特征和目标变量
X = data_imputed[['count','variance','range','max_diff','variance_diff','5th_diff_p',
          'mean_diff_p','count_type_1_or_3','variance_type_1_or_3','range_type_1_or_3',
          'min_diff_type_1_or_3','variance_diff_type_1_or_3','95th_diff_type_1_or_3',
          '5th_diff_type_1_or_3','mean_diff_type_1_or_3','variance_diff_p_type_1_or_3',
          'max_diff_p_type_1_or_3','min_diff_p_type_1_or_3','95th_diff_p_type_1_or_3',
          'mean_diff_p_type_1_or_3','iid_really_macderived_ratio','iid_embeddedipv4_32_ratio',
          'iid_random_ratio','iid_really_macderived_over_total_ratio']]
y = data_imputed['target']

# 再次处理 X 中的缺失值
if X.isnull().values.any():
    print("X still contains NaN values after imputation. Re-imputing...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# 创建 SVM 分类器实例
svm = SVC(probability=True)  # 设置 probability=True 以便获取概率估计
grid_search = GridSearchCV(svm, param_grid, refit=True, verbose=2, cv=5)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数的模型进行预测
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

# 打印分类报告
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
plot_learning_curve(best_svm, X_train_scaled, y_train, cv=5)
plt.xticks(fontsize=25)  # 增加x轴刻度字体大小
plt.yticks(fontsize=25)  # 增加y轴刻度字体大小

plt.tight_layout()  # 自动调整子图参数
plt.savefig('./png/SVM-new.png', dpi=400)
plt.show()
# 绘制 ROC 曲线
y_score = best_svm.decision_function(X_test_scaled)  # 获取决策函数值
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# 保存模型和标准化器
scaler_filename = './pkl/svm-scaler-96.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")

model_filename = './pkl/svm_model-929-all.pkl'
joblib.dump(best_svm, model_filename)
print(f"Best model saved to {model_filename}")