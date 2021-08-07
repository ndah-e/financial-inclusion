import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from categorical_encoder import CategoricalEncoder


class PrepareFeatures:
    def __init__(self, raw_data, target, test_size=0.2, cv=5):
        """"
            Initialise training data set and variables
            Param:
                raw_data: pandas dataframe
                target: target variable name (string)
                test_size: proportion of test size (float)
                cv: cross validation samples
        """

        self.raw_data = raw_data
        self.target = target
        self.test_size = test_size
        self.cv = cv
        self.prepared_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.continuous = []
        self.categorical = []
        self.discrete = []
        self.encoding_dict = {}

    def get_variable_types(self):
        """"
            Extract variable types form training data
        """
        for var in self.raw_data.columns:
            if self.war_data[var].dtype == "O":
                self.categorical.append(var)

        print("There are {} categorical variables".format(len(self.categorical)))

        # get all numeric variables
        numerical = []
        for var in self.raw_data.columns:
            if self.war_data[var].dtype != "O":
                numerical.append(var)

        # get discrete variables
        for var in numerical:
            if len(self.raw_data[var].unique()) < 20:
                self.discrete.append(var)

        print("There are {} discrete variables".format(len(self.discrete)))

        # get al continuous variables
        for var in numerical:
            if var not in self.discrete and var != self.target:
                self.continuous.append(var)
        print("There are {} numerical variables".format(len(numerical)))

    def split_data(self, *, training=False):
        """"
            if we are training set training = True, 
            then divide into train and test
        """
        if training:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.prepared_data,
                self.prepared_data[self.target],
                test_size=self.test_size,
                random_state=0,
            )

        print(
            "Number of samples in training set {}, test set {}".format(
                self.X_train.shape, self.X_test.shape
            )
        )

    def handle_missing_values(self):
        """"
            Impute missing values for continuous variables
            Add modality 'Missing'for categorical variables
        """
        for col in self.continuous:
            print("Code not implmented")

        # add new modality 'Missing' to categorical variables
        for var in self.categorical:
            self.prepared_data[var].fillna("Missing", inplace=True)

    def handle_rare_modalities(self, *, variable):
        """"
            Handle rare modalities in categorical variables
            Possible techniques to implement
                - merge with others selected sattistically
                - 
        """

        temp = self.raw_data.groupby([variable])[variable].count() / np.float(
            len(self.raw_data)
        )
        frequent_cat = [x for x in temp.loc[temp > 0.03].index.values]

        # replace the labels in data to be passed to model
        self.prepared_data[variable] = np.where(
            self.prepared_data[variable].isin(frequent_cat),
            self.prepared_data[variable],
            "Rare",
        )

    def prepare_data(self, training: bool = False):
        # This is the method where we capture  all the parameters during training,
        self.prepared_data = self.raw_data.copy(deep=True)

        self.separate_data_types()  # captures the different types of vars
        # self.handle_missing_values() # fills na
        # categorical_discrete = self.categorical + self.discrete
        # self.rare_imputation(variable=var) # replaces rare labels
        self.split_data(training=training)  # splits data fro training

        ce = CategoricalEncoder(raw_df=self.X_train, cv=self.cv)
        self.processed_data, self.encoding_dict = ce.categorical_encoding_rtt()

        if not training:
            if self.target in self.prepared_data.columns:
                self.prepared_data.drop([self.target], axis=1)
