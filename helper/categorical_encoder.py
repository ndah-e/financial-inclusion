import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
#import matplotlib.pyplot as plt


class CategoricalEncoder:
    """
    Methods to encode categorical variables to numeric variables
    eliminate the need for onehot encoding for categorical variables with many modalities
    """

    def __init__(self, raw_df: pd.DataFrame, target, cv=5, n=10):
        self.n = n  # int: maximum number of distinct values for a discrete variable
        self.cv = cv  # number of cross validation samples
        self.target = target  # string: name of target variable [target value 1 or 0]
        self.df = raw_df

        # global variables
        self.df_encoded = raw_df
        self.continuous = []
        self.categorical = []
        self.discrete = []
        self.binary = []

        # encoding dictionaries
        self.modality_count_dict = {}
        self.encoding_dict_binary = {}
        self.encoding_dict_cv = {}
        self.encoding_dict_average = {}

        self.separate_variable_types()

    def partition(self, list_in):
        random.shuffle(list_in)
        return [list_in[i :: self.cv] for i in range(self.cv)]

    def separate_variable_types(self):

        # find all binary variables
        for var in self.df.columns:
            if len(list(self.df[var].unique())) <= 2 and var != self.target:
                self.binary.append(var)
        print("There are {} binary variables".format(len(self.binary)))

        # find categorical variables
        for var in self.df.columns:
            if var in self.binary:
                continue
            if self.df[var].dtype == "O" and var != self.target:
                self.categorical.append(var)
        print("There are {} categorical variables".format(len(self.categorical)))

        # find all numerical variables
        numerical = []
        for var in self.df.columns:
            if var in self.binary:
                continue
            if self.df[var].dtype != "O":
                numerical.append(var)

        # find discrete variables (n max number of distinct values)
        for var in numerical:
            if var in self.binary:
                continue
            if len(self.df[var].unique()) <= self.n and var != self.target:
                self.discrete.append(var)
        print("There are {} discrete variables".format(len(self.discrete)))

        for var in numerical:
            if var in self.binary:
                continue
            if var not in self.discrete and var != self.target:
                self.continuous.append(var)
        print("There are {} continuous variables".format(len(self.continuous)))

        self.categorical = self.categorical + self.discrete
        print(
            "There are {} combined discrete + categorical variables".format(
                len(self.categorical)
            )
        )

    def rtt_encoding(self, data):
        """
            Fucntion to calculate the RTT (relation to target encoding)
            RTT: encodes variable using percentage of targets within each modality
            Index: normalize RTT by the proportion of targets in the data set.
        """
        df_cv = data.copy(deep=True)

        # target varibale counts
        target_counts_dict = df_cv[self.target].value_counts().to_dict()
        number_of_samples = sum(target_counts_dict.values())
        target_proportion = target_counts_dict[1] / number_of_samples

        # categorical variable counts
        self.modality_count_dict = {
            c: dict(df_cv[c].value_counts()) for c in self.categorical + self.binary
        }
        rtt_encoding_dict = {}
        for var in self.categorical + self.binary:
            rtt_encoding_dict[var] = {}
            rtt_min = 100
            rtt_max = 0
            for modality, count in self.modality_count_dict[var].items():
                targets_in_modality = df_cv[
                    (df_cv[var] == modality) & (df_cv[self.target] == 1)
                ].shape[0]
                rtt_value = targets_in_modality / count

                # normalize RTTs w.r.t. dataset
                rtt_value = rtt_value / target_proportion
                rtt_encoding_dict[var][modality] = rtt_value

                # check max rtt
                if rtt_value > rtt_max:
                    rtt_max = rtt_value

                # check rtt min
                if rtt_value < rtt_min:
                    rtt_min = rtt_value

            # min max normalize encoding
            # sum_over_all_rtt(rtt_normalized[i] * numberOfOccurencesPerModality[i]) /total_number_of_lines_minus_missing
            # TODO: account for range = 0
            rtt_range = rtt_max - rtt_min

            rtt_default = 0
            for modality in rtt_encoding_dict[var]:
                x_std = (rtt_encoding_dict[var][modality] - rtt_min) / rtt_range
                x_scaled = x_std * rtt_range + rtt_min
                rtt_encoding_dict[var][modality] = x_scaled

                # rtt_scaled_for default
                targets_in_modality = df_cv[
                    (df_cv[var] == modality) & (df_cv[self.target] == 1)
                ].shape[0]
                targets_in_modality = self.modality_count_dict[var][modality]
                rtt_default += x_scaled * targets_in_modality

            # default value for the variable
            rtt_encoding_dict[var]["default"] = rtt_default / df_cv.shape[0]

        return rtt_encoding_dict

    def categorical_encoding_rtt(self):
        # create cross validation samples
        index_list = self.df.index.to_list()
        folds = self.partition(index_list)

        for i in range(self.cv):
            fold = folds[i]
            df_tmp = self.df.loc[fold]
            self.encoding_dict_cv[i] = {}
            self.encoding_dict_cv[i] = self.rtt_encoding(df_tmp)

        for fold, encoding in self.encoding_dict_cv.items():
            for var in encoding:
                self.encoding_dict_average[var] = {}
                for modality, value in encoding[var].items():
                    if modality not in self.encoding_dict_average[var]:
                        self.encoding_dict_average[var][modality] = 0
                    self.encoding_dict_average[var][modality] += value / self.cv

        # binary encoding
        for var in self.binary:
            le = LabelEncoder().fit(self.df[var].values)
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            self.encoding_dict_binary[var] = mapping
            self.df_encoded[var] = self.df_encoded[var].apply(
                lambda x: self.encoding_dict_binary[var][x]
            )

        # categorical encoding
        for var in self.categorical:
            self.df_encoded[var] = self.df_encoded[var].apply(
                lambda x: self.encoding_dict_average[var][x]
                if x in self.encoding_dict_average[var]
                else self.encoding_dict_average[var]["default"]
            )

    # def plot_encoding(self):
    #     """Plot index on all variables"""
    #     for var in self.encoding_dict_average:

    #         plt.subplots(figsize=(10, 6))
    #         x_axis = list(ce.modality_count_dict[var].keys())
    #         y_axis = [ce.modality_count_dict[var][modality] for modality in x_axis]
    #         y_axis_index = [
    #             ce.encoding_dict_average[var][modality] for modality in x_axis
    #         ]

    #         ax = sns.barplot(x=x_axis, y=y_axis)
    #         ax.set_title(var)
    #         ax.set_xticklabels(labels=x_axis, rotation=30, ha="right")
    #         ax2 = ax.twinx()
    #         ax2.plot(ax.get_xticks(), y_axis_index, lw=2, color="black")
    #         plt.show()
