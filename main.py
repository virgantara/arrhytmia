import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,KFold, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report, f1_score, \
    recall_score, cohen_kappa_score, precision_score
from ELM import ELM
import matplotlib.pyplot as plt
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from pprint import pprint
from sklearn.decomposition import PCA
import seaborn as sn

le = preprocessing.LabelEncoder()

enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
std_sc = preprocessing.StandardScaler()
sc = preprocessing.MinMaxScaler()
pca = PCA(n_components=5)

headers = ["atr"+str(i) for i in range(1,30)]
headers = np.append(headers,'class')
# print(headers)
data = pd.read_csv("features_ensembled.csv",names=headers)
# data = pd.read_csv("dataset/mfcc_features_imbalance_scaled.csv")
# print(data.head())

def calc_performance(yt, yp):
    acc = accuracy_score(y_true=yt, y_pred=yp)
    f1 = f1_score(y_true=yt, y_pred=yp)
    rec = recall_score(y_true=yt, y_pred=yp)
    prec = precision_score(y_true=yt, y_pred=yp)
    auc = roc_auc_score(y_true=yt, y_score=yp)
    cohen = cohen_kappa_score(y1=yt, y2=yp)

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC' : auc,
        'Cohen' : cohen
    }

def get_result_cross_validation(model_nama, jumlah_fold, X, y):
    if model_nama is 'NB':
        model = MultinomialNB()
    elif model_nama is 'DT':
        model = DecisionTreeClassifier()
    elif model_nama is 'RF':
        model = RandomForestClassifier(n_estimators=10)
    elif model_nama is 'XGB':
        model = GradientBoostingClassifier()
    elif model_nama is 'LR':
        model = LogisticRegression()
    elif model_nama is 'ADA':
        model = AdaBoostClassifier()
    elif model_nama is 'SVM':
        model = SVC(
            kernel='rbf',
            gamma='auto'
        )

    cv = KFold(n_splits=jumlah_fold, random_state=42, shuffle=True)
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        clf = model.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        scores.append(clf.score(X_test, y_test))

    return scores
        # hasil = calc_performance(y_test, y_pred)

def get_result_classifier(model_nama, X_train, X_test, y_train, y_test):
    if model_nama is 'NB':
        model = MultinomialNB()
    elif model_nama is 'DT':
        model = DecisionTreeClassifier()
    elif model_nama is 'RF':
        model = RandomForestClassifier(n_estimators=10)
    elif model_nama is 'XGB':
        model = GradientBoostingClassifier()
    elif model_nama is 'LR':
        model = LogisticRegression()
    elif model_nama is 'ADA':
        model = AdaBoostClassifier()
    elif model_nama is 'SVM':
        model = SVC(
            kernel='rbf',
            gamma='auto'
        )

    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # class_report = classification_report(y_test, y_pred, zero_division=0)
    # print(model_nama,":",class_report)
    return calc_performance(y_test, y_pred)

def get_result_elm(data_path):
    data = pd.read_csv(data_path)
    X = data.iloc[:, 0:13]
    y = data.iloc[:, 13]


    # y = data['variety'].astype("category")
    #
    # X = data.drop('variety', axis=1)
    y = np.array(y).reshape(-1, 1)
    y_encoded = enc.fit_transform(y).toarray().astype(np.float32)
    n_classes = y_encoded.shape[1]

    normalized_X = preprocessing.normalize(X)

    x_train, x_test, t_train, t_test = train_test_split(normalized_X, y_encoded, test_size=0.4, random_state=42,
                                                        shuffle=True)

    model = ELM(
        jumlah_input_nodes=X.shape[1],
        jumlah_hidden_node=14,
        jumlah_output=n_classes,
    )

    model.fit(x_train, t_train)
    yp = model.feedforward(x_test)
    yt = t_test
    y_pred = np.argmax(yp, axis=-1)
    y_true = np.argmax(yt, axis=-1)
    return calc_performance(y_true, y_pred)

def get_data(dataset):

    X = dataset.iloc[:, 0:13]
    y = dataset.iloc[:, 13]

    y = le.fit_transform(y)

    # X = data.drop(['class'],axis=1)
    # y = data['class']
    # print(X.head())
    # y = data['variety'].astype("category")
    #
    # X = data.drop('variety', axis=1)
    # y = np.array(y).reshape(-1, 1)
    # n_classes = 2

    normalized_X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.4, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test

# data.groupby('class').size().plot(kind='pie',
#                                        y = "Class",
#                                        label = "Type",
#                                        autopct='%1.1f%%')
# plt.show()
def get_data_from_downsampling(dataset):
    normal_hb = dataset[dataset['class'] == "Normal"]
    abnormal_hb = dataset[dataset['class'] == "Abnormal"]

    normal_downsample = resample(normal_hb,replace=True,n_samples=len(abnormal_hb),random_state=42)

    concated = pd.concat([normal_downsample, abnormal_hb])

    X = concated.drop('class', axis=1)
    y = concated['class']
    y = le.fit_transform(y)

    normalized_X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.4, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test

def get_data_from_upsampling(dataset):
    normal_hb = dataset[dataset['class'] == "Normal"]
    abnormal_hb = dataset[dataset['class'] == "Abnormal"]

    abnormal_upsample = resample(abnormal_hb,replace=True,n_samples=len(normal_hb),random_state=42)

    concated = pd.concat([abnormal_upsample, normal_hb])
    X = concated.drop('class', axis=1)
    y = concated['class']
    y = le.fit_transform(y)

    normalized_X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.4, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test
def get_data_from_smote(dataset, is_pca=False, n_component=5):
    smt = SMOTE(random_state=42)
    X = dataset.drop('class',axis=1)
    y = dataset['class']
    X_smoted, y_smoted = smt.fit_resample(X, y)
    y = le.fit_transform(y_smoted)

    if is_pca:
        pca = PCA(n_components=n_component)
        X_smoted = pca.fit_transform(X_smoted)

    normalized_X = sc.fit_transform(X_smoted)

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.4, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test


def get_optimized_model(is_using_optimized=True):
    if is_using_optimized:
        hasil = RandomForestClassifier(
            bootstrap=False,
            max_depth=30,
            max_features='sqrt',
            min_samples_leaf=2,
            min_samples_split=2,
            n_estimators=800
        )
    else:
        rf = RandomForestClassifier()
        num_estimator = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt', 'log2']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)

        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {
            'n_estimators': num_estimator,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }

        hasil = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=100,
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

    return hasil

def get_optimized_model_grid(is_using_optimized=True):
    if is_using_optimized:
        hasil = RandomForestClassifier(
            bootstrap=True,
            max_depth=100,
            max_features='auto',
            min_samples_leaf=2,
            min_samples_split=2,
            n_estimators=200
        )
    else:
        rf = RandomForestClassifier()
        num_estimator = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt', 'log2']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)

        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        param_grid = {
            'n_estimators': num_estimator,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }

        hasil = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            verbose=2,
            n_jobs=-1
        )

    return hasil


# downsampling
# data_prep = get_data_from_downsampling(data)
#
# # upsampling
# data_prep = get_data_from_upsampling(data)
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)
    # mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 * accuracy_score(test_labels, predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    perf = calc_performance(test_labels, predictions)
    print(perf)
    return accuracy

def get_optimized_model_mlp(is_using_optimized=True):
    if is_using_optimized:
        hasil = MLPClassifier(
            solver='adam',
            learning_rate_init=0.002,
            learning_rate='constant',
            hidden_layer_sizes=(50,100,50),
            activation='relu'
        )
    else:
        rf = MLPClassifier()

        params = {
            'hidden_layer_sizes': [(10,10,10),(20,20,20),(50,50,50), (50,100,50), (10,),(20,),(50,),(100,),(10,10),(20,20),(50,50),(100,100)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            # 'max_iter' : 300,
            # 'alpha': [0.001, 0.05],
            'learning_rate_init':[0.001,0.002,0.003],
            'learning_rate': ['constant','adaptive'],
        }

        hasil = RandomizedSearchCV(
            estimator=rf,
            param_distributions=params,
            n_iter=100,
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

    return hasil

data_prep = get_data_from_smote(data, is_pca=False,n_component=16)

X_train, X_test, y_train, y_test = data_prep

# rf_random = get_optimized_model(is_using_optimized=False)
is_using_optimized = False
mlp_random = get_optimized_model_mlp(is_using_optimized=is_using_optimized)
# rf_grid = get_optimized_model_grid(is_using_optimized=False)

mlp_random.fit(X_train, y_train)
# print(rf_random.best_params_)

# rf_grid.fit(X_train, y_train)
# print(rf_grid.best_params_)

#
base_model = MLPClassifier(hidden_layer_sizes=(10,10),learning_rate='constant',learning_rate_init=0.01,solver='adam')
# base_model = RandomForestClassifier(n_estimators=10, random_state=42)
base_model.fit(X_train, y_train)

base_acc = evaluate(base_model, X_test, y_test)
random_acc = evaluate(mlp_random, X_test, y_test)
# grid_acc = evaluate(rf_grid, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_acc - base_acc) / base_acc))
if not is_using_optimized:
    print(mlp_random.best_params_)
# print('Improvement of {:0.2f}%.'.format( 100 * (grid_acc - base_acc) / base_acc))

# print(normal_downsample.shape)
# # hasil_elm = get_result_elm("dataset/mfcc_features_imbalance.csv")

hasil_nb = get_result_classifier('NB',X_train, X_test, y_train, y_test)
hasil_dt = get_result_classifier('DT',X_train, X_test, y_train, y_test)
hasil_rf = get_result_classifier('RF',X_train, X_test, y_train, y_test)
hasil_xgb = get_result_classifier('XGB',X_train, X_test, y_train, y_test)
hasil_ada = get_result_classifier('ADA',X_train, X_test, y_train, y_test)
hasil_lr = get_result_classifier('LR',X_train, X_test, y_train, y_test)
hasil_svm = get_result_classifier('SVM',X_train, X_test, y_train, y_test)
#
print("Naive Bayes: ",hasil_nb)
print("Decision Tree: ",hasil_dt)
print("Random Forest: ",hasil_rf)
print("XGB: ",hasil_xgb)
print("ADA: ",hasil_ada)
print("LR", hasil_lr)
print("SVM", hasil_svm)
# # print("ELM", hasil_elm)

# hasil_cv = get_result_cross_validation('NB',10,normalized_X,y)
# print('Hasil NB CV:',hasil_cv)
#
# hasil_cv = get_result_cross_validation('DT',10,normalized_X,y)
# print('Hasil DT CV:',hasil_cv)
#
# hasil_cv = get_result_cross_validation('RF',10,normalized_X,y)
# print('Hasil RF CV:',hasil_cv)

# Plot correlation
# df = pd.DataFrame(normalized_X)
# corrMatrix = df.corr()
# sn.heatmap(corrMatrix, annot=True)
# plt.show()
# print(data.i[12])
# print(data['class'].value_counts())

