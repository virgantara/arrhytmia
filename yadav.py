import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
sc = preprocessing.MinMaxScaler()
pca = PCA(n_components=16)


def calc_performance_multiclass(yt, yp):
    acc = accuracy_score(y_true=yt, y_pred=yp)
    f1 = f1_score(y_true=yt, y_pred=yp, average='weighted')
    rec = recall_score(y_true=yt, y_pred=yp,average='weighted')
    prec = precision_score(y_true=yt, y_pred=yp,average='weighted')
    # auc = roc_auc_score(y_true=yt, y_score=yp,multi_class='ovr')
    cohen = cohen_kappa_score(y1=yt, y2=yp)

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        # 'AUC' : auc,
        'Cohen' : cohen
    }

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

def get_result_classifier(model_nama, X_train, X_test, y_train, y_test):
    if model_nama is 'NB':
        model = GaussianNB()
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
    elif model_nama is 'KNN':
        model = KNeighborsClassifier(n_neighbors=5,weights='distance')
    elif model_nama is 'SVM':
        model = SVC(
            kernel='linear',
            gamma='auto'
        )

    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # class_report = classification_report(y_test, y_pred, zero_division=0)
    # print(model_nama,":",class_report)
    return calc_performance_multiclass(y_test, y_pred)


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


def run_yadav(X_train, X_test, y_train, y_test):

    hasil_svm = get_result_classifier('SVM', X_train, X_test, y_train, y_test)
    hasil_nb = get_result_classifier('NB',X_train, X_test, y_train, y_test)
    hasil_rf = get_result_classifier('RF',X_train, X_test, y_train, y_test)
    hasil_knn = get_result_classifier('KNN', X_train, X_test, y_train, y_test)

    print("SVM", hasil_svm)
    print("Naive Bayes: ",hasil_nb)
    print("Random Forest: ",hasil_rf)
    print("KNN: ", hasil_knn)

def run_our(X_train, X_test, y_train, y_test):
    is_using_optimized = True
    mlp_random = get_optimized_model_mlp(is_using_optimized=is_using_optimized)
    # rf_grid = get_optimized_model_grid(is_using_optimized=False)

    mlp_random.fit(X_train, y_train)
    # print(rf_random.best_params_)

    # rf_grid.fit(X_train, y_train)
    # print(rf_grid.best_params_)

    #
    base_model = MLPClassifier(hidden_layer_sizes=(10, 10), learning_rate='constant', learning_rate_init=0.01,
                               solver='adam')
    # base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    base_model.fit(X_train, y_train)

    base_acc = evaluate(base_model, X_test, y_test)
    random_acc = evaluate(mlp_random, X_test, y_test)
    # grid_acc = evaluate(rf_grid, X_test, y_test)
    print('Improvement of {:0.2f}%.'.format(100 * (random_acc - base_acc) / base_acc))
    if not is_using_optimized:
        print(mlp_random.best_params_)

    hasil_svm = get_result_classifier('SVM', X_train, X_test, y_train, y_test)
    hasil_nb = get_result_classifier('NB', X_train, X_test, y_train, y_test)
    hasil_rf = get_result_classifier('RF', X_train, X_test, y_train, y_test)
    hasil_knn = get_result_classifier('KNN', X_train, X_test, y_train, y_test)
    print("\nOthers")
    print("SVM", hasil_svm)
    print("Naive Bayes: ", hasil_nb)
    print("Random Forest: ", hasil_rf)
    print("KNN: ", hasil_knn)

headers = ["atr"+str(i) for i in range(1,10)]
headers = np.append(headers,'class')
# print(headers)
data = pd.read_csv("yadav_features_yaseen_data.csv",names=headers)

data_prep = get_data_from_smote(data, is_pca=False)

X_train, X_test, y_train, y_test = data_prep
print("Hasil Yadav")
print("========================================================")
run_yadav(X_train, X_test, y_train, y_test)
#
# headers = ["atr"+str(i) for i in range(1,56)]
# headers = np.append(headers,'class')
# # print(headers)
# data = pd.read_csv("features_ensemble.csv",names=headers)
#
# data_prep = get_data_from_smote(data, is_pca=True,n_component=16)
#
# X_train, X_test, y_train, y_test = data_prep
#
# print("Hasil Our")
# print("========================================================")
# run_yadav(X_train, X_test, y_train, y_test)
# rf_random = get_optimized_model(is_using_optimized=False)

# print('Improvement of {:0.2f}%.'.format( 100 * (grid_acc - base_acc) / base_acc))

# print(normal_downsample.shape)
# # hasil_elm = get_result_elm("dataset/mfcc_features_imbalance.csv")
#


# #
# print("Naive Bayes: ",hasil_nb)
# print("Decision Tree: ",hasil_dt)
# print("Random Forest: ",hasil_rf)
# print("XGB: ",hasil_xgb)
# print("ADA: ",hasil_ada)
# print("LR", hasil_lr)
# print("SVM", hasil_svm)
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

