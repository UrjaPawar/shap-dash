from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

class Data:
    heart_path = "data/uci_heart.csv"
    cerv_path = "data/risk_factors_cervical_cancer.csv"
    diab_path = "data/diabetes.csv"

    def __init__(self, dataset_name, one_hot_encode= True, test_split = 0.2, shuffle_while_training = True, random_state = 24, pick_without_insulin=True):
        self.df = pd.DataFrame()
        self.dec_precisions = {}
        self.changes_allowed = {}
        self.cluster_order = []
        self.features = []
        self.target = []
        self.categorical = []
        self.one_hot = []
        self.continuous = []
        self.binary = []
        self.nick_name_dict = {}
        self.encoded_df = None
        self.changes_count_dict = {}
        self.feature_range = {}
        if dataset_name == "Heart DB":
            self.load_heart()
        elif dataset_name == "Cervical DB":
            self.load_cervical()
        elif dataset_name == "Diabetes DB":
            self.load_diabetes(pick_without_insulin)
        if one_hot_encode:
            self.one_hot_encode()
        else:
            self.encoded_df = self.df

        # self.split_data(test_split, shuffle_while_training, random_state)


    def remove_missing_vals(self, filepath, missing_arr):
        print("Missing values will be removed")
        df = pd.read_csv(filepath, na_values=missing_arr)
        df = df.dropna()
        return df

    def split_data(self, test_split, shuffle, random_state):
        print("Splitting into training and testing sets")
        # explore the stratify option
        X_train, X_test, y_train, y_test = train_test_split(self.encoded_df.drop(self.target, axis=1), self.df[self.target], test_size=test_split,
                                                            random_state=random_state, shuffle = shuffle)
        self.train_df = pd.concat([X_train, y_train], axis=1)
        self.test_df = pd.concat([X_test, y_test], axis=1)


    def load_heart(self):
        missing_values = ["?"]
        data = self.remove_missing_vals(self.heart_path, missing_values)

        data['Typical_Angina'] = (data['Chest_Pain'] == 1).astype(int)
        data['Atypical_Angina'] = (data['Chest_Pain'] == 2).astype(int)
        data['Asymptomatic_Angina'] = (data['Chest_Pain'] == 4).astype(int)
        data['Non_Anginal_Pain'] = (data['Chest_Pain'] == 3).astype(int)
        data = data.drop(columns=['Chest_Pain'])
        data["MHR"] = 220 - data["Age"]
        data["mhr_exceeded"] = data["MHR"] < data["MAX_Heart_Rate"]
        data["mhr_exceeded"] = data["mhr_exceeded"].astype(int)
        data = data.drop(columns=["MAX_Heart_Rate", "MHR"])
        # self.features = ['Age', 'Sex', 'Typical_Angina', 'Atypical_Angina', 'Asymptomatic_Angina', 'Non_Anginal_Pain','Resting_Blood_Pressure',
        #                    'Fasting_Blood_Sugar', 'Rest_ECG', 'Colestrol',
        #                    'Slope', 'ST_Depression', 'Exercised_Induced_Angina', 'mhr_exceeded', 'Major_Vessels',
        #                    'Thalessemia']
        self.features = ['Age', 'Sex', 'Typical_Angina', 'Atypical_Angina', 'Resting_Blood_Pressure',
                    'Fasting_Blood_Sugar', 'Rest_ECG', 'Colestrol', 'Non_Anginal_Pain',
                    'Slope', 'ST_Depression', 'Asymptomatic_Angina', 'Exercised_Induced_Angina',
                    'mhr_exceeded', 'Major_Vessels','Thalessemia']
        nick_names = ["Age", "Sex", "TA", "ATA", "BP", "Sugar", "ECG", "Ch", "NAP", "Sl", "ST", "AA",
                      "EIA", "MHR", "MV", "Thal"]

        for ind,feat in enumerate(self.features):
            self.nick_name_dict[feat] = nick_names[ind]


        TARGET_COLUMNS = ['Target']
        data[TARGET_COLUMNS] = data[TARGET_COLUMNS] != 0
        data[TARGET_COLUMNS] = data[TARGET_COLUMNS].astype(int)
        self.continuous = ['Age', 'Colestrol', 'Resting_Blood_Pressure','ST_Depression']
        self.target = TARGET_COLUMNS[0]
        self.df = data[self.features + [self.target]]
        self.binary = ['Sex', 'Typical_Angina', 'Atypical_Angina', 'Fasting_Blood_Sugar', 'Asymptomatic_Angina', 'Non_Anginal_Pain',
                       'Exercised_Induced_Angina', 'mhr_exceeded']
        self.dec_precisions = {'Age':0, 'Resting_Blood_Pressure':0, 'Colestrol':0, 'ST_Depression':1}
        self.cluster_order = [['Age', 'Sex', 'Colestrol',  'Asymptomatic_Angina', 'Major_Vessels','Thalessemia'],
                              ['Age', 'Sex'],
                              ['Age', 'Sex', 'Colestrol'],
                              ['Age', 'Sex', 'Asymptomatic_Angina', 'Major_Vessels', 'Thalessemia'],
                              ['Age', 'Sex', 'Thalessemia'],
                              ['Age', 'Sex', 'Major_Vessels']]
        # 5,11,
        self.changes_count_dict = {'Age': 5, 'Sex': {(0, 0): 1, (1, 1): 1, (0, 1): 1},
                              'Typical_Angina': {(0, 0): 1, (1, 1): 1, (0, 1): 1},
                              'Atypical_Angina': {(0, 0): 1, (1, 1): 1, (0, 1): 1},
                              'Resting_Blood_Pressure': 11,
                              'Fasting_Blood_Sugar': {(0, 0): 1, (1, 1): 1, (0, 1): 1},
                              'Rest_ECG': {(0, 1): 1, (1, 2): 1, (0, 2): 2, (1, 1): 1, (2, 2): 1, (0, 0): 1},
                              'Colestrol': 11, 'Asymptomatic_Angina': {(0, 0): 1, (1, 1): 1, (0, 1): 1},
                              'Non_Anginal_Pain': {(0, 0): 1, (1, 1): 1, (0, 1): 1},
                              'Slope': {(2, 1): 1, (1, 3): 2, (3, 2): 1, (1, 1): 1, (2, 2): 1, (3, 3): 1},
                              'ST_Depression': 0.3, 'Exercised_Induced_Angina': {(0, 0): 1, (1, 1): 1, (0, 1): 1},
                              'mhr_exceeded': {(0, 0): 1, (1, 1): 1, (0, 1): 1},
                              'Major_Vessels': {(0, 1): 1, (1, 2): 1, (2, 3): 1, (0, 2): 2, (1, 3): 2, (0, 3):
                                  3, (1, 1): 1, (2, 2): 1, (3, 3): 1, (0, 0): 1},
                              'Thalessemia': {(6, 7): 2, (6, 3): 2, (3, 7): 1, (7, 7): 1, (6, 6): 1, (3, 3): 1}}

        self.categorical = [feature for feature in self.features if feature not in self.continuous]
        self.feature_range = {feat: list(set(self.df[feat])) for feat in self.categorical}


    def load_diabetes(self, pick_without_insulin):
        df = pd.read_csv(self.diab_path)
        self.features = list(df.columns)
        if pick_without_insulin:
            self.df = df.drop(columns=['Insulin'])
            self.features.remove('Insulin')
        else:
            self.df = df[df['Insulin']>0]
        self.features.remove('Outcome')
        self.target = "Outcome"
        self.continuous = self.features.copy()
        self.dec_precisions = {
            'Pregnancies':0,
            'Glucose':0,
            'BloodPressure':0,
            'SkinThickness':0,
            'BMI':1,
            'DiabetesPedigreeFunction':3,
            'Age':0
        }
        self.changes_count_dict = {
            'Pregnancies': 1,
            'Glucose': 8,
            'BloodPressure': 5,
            'SkinThickness': 0,
            'BMI': 2,
            'DiabetesPedigreeFunction': 0.08,
            'Age': 5
        }


    def load_cervical(self):
        df = pd.read_csv(self.cerv_path, na_values=["?"])
        numerical_s = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies',
                     'Hormonal Contraceptives (years)',
                     'IUD (years)', 'STDs (number)', 'STDs: Number of diagnosis', 'Smokes (packs/year)',
                     'Smokes (years)']
        self.dec_precisions = {
            'Age':0, 'Number of sexual partners':0, 'First sexual intercourse':0, 'Num of pregnancies':0,
                     'Hormonal Contraceptives (years)':0,
                     'IUD (years)':0, 'STDs (number)':0, 'STDs: Number of diagnosis':0, 'Smokes (packs/year)':0,
                     'Smokes (years)':0
        }
        self.changes_allowed = {
            'Age':5, 'Number of sexual partners':1, 'First sexual intercourse':1, 'Num of pregnancies':1,
                     'Hormonal Contraceptives (years)':1,
                     'IUD (years)':1, 'STDs (number)':1, 'STDs: Number of diagnosis':1, 'Smokes (packs/year)':1,
                     'Smokes (years)':1
        }

        # not sure why I wanted to drop these columns
        # df = df.drop(
        # columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 'Hormonal Contraceptives',
        #          'IUD', 'Smokes', 'STDs', 'STDs:HIV',
        #          'STDs:condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:genital herpes','STDs:cervical condylomatosis','STDs:AIDS'])
        df = df.drop(
        columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
                 'STDs:condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:genital herpes','STDs:cervical condylomatosis'])
        df = df.dropna()
        tar = 'Biopsy'
        feats = list(df.columns)
        feats.remove(tar)

        self.continuous = [feat for feat in feats if feat in numerical_s]
        self.categorical = df[feats].columns.difference(self.continuous)

        ada = ADASYN(random_state=28)
        X_ada, y_ada = ada.fit_resample(df[feats], df[tar])
        m2a = pd.DataFrame(X_ada)
        m2b = pd.DataFrame(y_ada)
        for feat in self.continuous:
            m2a[feat] = m2a[feat].round(self.dec_precisions[feat])
        for feat in self.categorical:
            m2a[feat] = m2a[feat].round(0)
        m2b.columns = ['Biopsy']
        self.df = m2a.join(m2b)
        self.target = 'Biopsy'
        self.changes_count_dict = {'Age': 5, 'Number of sexual partners': 2, 'First sexual intercourse': 3, 'Num of pregnancies': 1,
                     'Hormonal Contraceptives (years)': 1,
                     'IUD (years)': 1, 'STDs (number)': 1, 'STDs: Number of diagnosis':1, 'Smokes (packs/year)':1,
                     'Smokes (years)': 1,
                                   'Citology': {(0, 0): 1, (1, 1): 1, (0, 1): 0}, 'Dx': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                   'Dx:CIN': {(0, 0): 1, (1, 1): 1, (0, 1): 0}, 'Dx:Cancer': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                   'Dx:HPV': {(0, 0): 1, (1, 1): 1, (0, 1): 0}, 'Hinselmann': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                  'Hormonal Contraceptives': {(0, 0): 1, (1, 1): 1, (0, 1): 0}, 'IUD': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                   'STDs': {(0, 0): 1, (1, 1): 1, (0, 1): 0}, 'STDs:AIDS': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                   'STDs:HIV': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                  'STDs:HPV': {(0, 0): 1, (1, 1): 1, (0, 1): 0}, 'STDs:Hepatitis B': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                   'STDs:molluscum contagiosum': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                  'STDs:pelvic inflammatory disease': {(0, 0): 1, (1, 1): 1, (0, 1): 0}, 'STDs:syphilis': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                  'STDs:vaginal condylomatosis': {(0, 0): 1, (1, 1): 1, (0, 1): 0},
                                  'Schiller': {(0, 0): 1, (1, 1): 1, (0, 1): 0}, 'Smokes': {(0, 0): 1, (1, 1): 1, (0, 1): 0}}
        self.features = list(self.df.columns)
        self.features.remove(self.target)
        self.feature_range = {feat: list(set(self.df[feat])) for feat in self.categorical}
        self.feature_range['STDs:AIDS'] = [0,1]

    def one_hot_encode(self):
        self.encoded_df = pd.get_dummies(self.df, columns=self.categorical)
        self.features = list(self.encoded_df.columns)
        self.features.remove(self.target)



