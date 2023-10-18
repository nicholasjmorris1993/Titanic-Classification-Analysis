def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, KBinsDiscretizer, PolynomialFeatures
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from scipy.stats import kstest
from itertools import combinations
import pandas_datareader as pdr
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

if os.name == "nt":
    path_sep = "\\"
else:
    path_sep = "/"


class Classification:
    def __init__(
        self, 
        name="Classification Analysis", 
        path=None,
        rename=True, 
        time=True, 
        binary=True, 
        imputation=True, 
        variance=True,
        scale=True,
        atwood=True,
        binning=True,
        reciprocal=True, 
        interaction=True, 
        selection=True,
        plots=True,
    ):
        self.name = name  # name of the analysis
        self.path = path  # the path where results will be exported
        self.rename = rename  # should features be renamed to remove whitespace?
        self.time = time  # should datetime features be computed?
        self.binary = binary  # should categorical features be converted to binary features?
        self.imputation = imputation  # should missing values be filled in?
        self.variance = variance  # should we remove constant features?
        self.scale = scale  # should we scale the features?
        self.atwood = atwood  # should we compute atwood numbers?
        self.binning = binning  # should we put continous features into bins?
        self.reciprocal = reciprocal  # should reciporcals be computed?
        self.interaction = interaction  # should interactions be computed?
        self.selection = selection  # should we perform feature selection?
        self.plots = plots  # should we plot the analysis?

        if self.path is None:
            self.path = os.getcwd()

        # create folders for output files
        self.folder(f"{self.path}{path_sep}{self.name}")
        self.folder(f"{self.path}{path_sep}{self.name}{path_sep}dump")  # machine learning pipeline and data
        if self.plots:
            self.folder(f"{self.path}{path_sep}{self.name}{path_sep}plots")  # html figures

    def validate(self, X, y):
        # raw data
        self.X = X.copy()
        self.y = y.copy()

        # encode labels for the target
        self.labeler = LabelEncoder()
        y.iloc[:, 0] = self.labeler.fit_transform(y.iloc[:, 0].tolist())
        y = y.astype(int)

        # split up the data into training and testing
        trainX = X.head(int(0.8 * X.shape[0]))
        trainy = y.head(int(0.8 * y.shape[0]))
        testX = X.tail(int(0.2 * X.shape[0])).reset_index(drop=True)
        testy = y.tail(int(0.2 * y.shape[0])).reset_index(drop=True)

        print("Model Training:")
        start = time.time()

        # set up the machine learning pipeline
        self.names1 = FeatureNames(self.rename)
        self.datetime = TimeFeatures(self.time)
        self.categorical = CategoricalFeatures(self.binary)
        self.names2 = FeatureNames(self.rename)
        self.impute = ImputeFeatures(self.imputation)
        self.constant1 = ConstantFeatures(self.variance)
        self.selection1 = FeatureSelector(self.selection)
        self.scaler1 = ScaleFeatures(self.atwood or self.reciprocal, bounds=(0.2, 0.8))
        self.numbers = AtwoodNumbers(self.atwood)
        self.bin = BinFeatures(self.binning)
        self.reciprocals = Reciprocals(self.reciprocal)
        self.interactions = Interactions(self.interaction)
        self.constant2 = ConstantFeatures(self.reciprocal and self.interaction)
        self.selection2 = FeatureSelector(self.selection and (self.atwood or self.binning or self.reciprocal or self.interaction))
        self.scaler2 = ScaleFeatures(self.scale, bounds=(0, 1))
        self.lasso = LogisticRegressionCV(
            penalty="l1", 
            solver="saga",
            Cs=16, 
            cv=3, 
            tol=1e-4,
            max_iter=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # run the pipeline on training data
        print("> Transforming The Training Data")
        trainX = self.names1.fit_transform(trainX)
        trainX = self.datetime.fit_transform(trainX)
        trainX = self.categorical.fit_transform(trainX)
        trainX = self.names2.fit_transform(trainX)
        trainX = self.impute.fit_transform(trainX)
        trainX = self.constant1.fit_transform(trainX)
        trainX = self.selection1.fit_transform(trainX, trainy)
        trainX = self.scaler1.fit_transform(trainX)
        numbers = self.numbers.fit_transform(trainX)
        trainX = self.bin.fit_transform(trainX)
        trainX = self.reciprocals.fit_transform(trainX)
        trainX = self.interactions.fit_transform(trainX)
        trainX = pd.concat([trainX, numbers], axis="columns")
        trainX = self.constant2.fit_transform(trainX)
        trainX = self.selection2.fit_transform(trainX, trainy)
        trainX = self.scaler2.fit_transform(trainX)
        print("> Training Lasso")
        self.lasso.fit(trainX, trainy)

        end = time.time()
        self.run_time(start, end)

        print("Model Performance:")
        start = time.time()

        # transform the testing data and score the performance
        print("> Transforming The Testing Data")
        testX = self.names1.transform(testX)
        testX = self.datetime.transform(testX)
        testX = self.categorical.transform(testX)
        testX = self.names2.transform(testX)
        testX = self.impute.transform(testX)
        testX = self.constant1.transform(testX)
        testX = self.selection1.transform(testX)
        testX = self.scaler1.transform(testX)
        numbers = self.numbers.transform(testX)
        testX = self.bin.transform(testX)
        testX = self.reciprocals.transform(testX)
        testX = self.interactions.transform(testX)
        testX = pd.concat([testX, numbers], axis="columns")
        testX = self.constant2.transform(testX)
        testX = self.selection2.transform(testX)
        testX = self.scaler2.transform(testX)
        print("> Scoring The Model")
        self.performance(testX, testy)

        end = time.time()
        self.run_time(start, end)

        print("Model Indicators:")
        start = time.time()

        print("> Extracting Important Features")
        self.importance(trainX)

        end = time.time()
        self.run_time(start, end)

    def fit(self, X, y):
        # raw data
        self.X = X.copy()
        self.y = y.copy()

        # encode labels for the target
        self.labeler = LabelEncoder()
        y.iloc[:, 0] = self.labeler.fit_transform(y.iloc[:, 0].tolist())
        y = y.astype(int)

        print("Model Training:")
        start = time.time()

        # set up the machine learning pipeline
        self.names1 = FeatureNames(self.rename)
        self.datetime = TimeFeatures(self.time)
        self.categorical = CategoricalFeatures(self.binary)
        self.names2 = FeatureNames(self.rename)
        self.impute = ImputeFeatures(self.imputation)
        self.constant1 = ConstantFeatures(self.variance)
        self.selection1 = FeatureSelector(self.selection)
        self.scaler1 = ScaleFeatures(self.atwood or self.reciprocal, bounds=(0.2, 0.8))
        self.numbers = AtwoodNumbers(self.atwood)
        self.bin = BinFeatures(self.binning)
        self.reciprocals = Reciprocals(self.reciprocal)
        self.interactions = Interactions(self.interaction)
        self.constant2 = ConstantFeatures(self.reciprocal and self.interaction)
        self.selection2 = FeatureSelector(self.selection and (self.atwood or self.binning or self.reciprocal or self.interaction))
        self.scaler2 = ScaleFeatures(self.scale, bounds=(0, 1))
        self.lasso = LogisticRegressionCV(
            penalty="l1", 
            solver="saga",
            Cs=16, 
            cv=3, 
            tol=1e-4,
            max_iter=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        
        # run the pipeline on the data
        print("> Transforming The Data")
        X = self.names1.fit_transform(X)
        X = self.datetime.fit_transform(X)
        X = self.categorical.fit_transform(X)
        X = self.names2.fit_transform(X)
        X = self.impute.fit_transform(X)
        X = self.constant1.fit_transform(X)
        X = self.selection1.fit_transform(X, y)
        X = self.scaler1.fit_transform(X)
        numbers = self.numbers.fit_transform(X)
        X = self.bin.fit_transform(X)
        X = self.reciprocals.fit_transform(X)
        X = self.interactions.fit_transform(X)
        X = pd.concat([X, numbers], axis="columns")
        X = self.constant2.fit_transform(X)
        X = self.selection2.fit_transform(X, y)
        X = self.scaler2.fit_transform(X)
        print("> Training Lasso")
        self.lasso.fit(X, y)

        end = time.time()
        self.run_time(start, end)

        print("Model Indicators:")
        start = time.time()

        print("> Extracting Important Features")
        self.importance(X)

        end = time.time()
        self.run_time(start, end)
                    
    def predict(self, X):
        print("Model Prediction:")
        start = time.time()

        # transform and predict new data
        print("> Transforming The New Data")
        X = self.names1.transform(X)
        X = self.datetime.transform(X)
        X = self.categorical.transform(X)
        X = self.names2.transform(X)
        X = self.impute.transform(X)
        X = self.constant1.transform(X)
        X = self.selection1.transform(X)
        X = self.scaler1.transform(X)
        numbers = self.numbers.transform(X)
        X = self.bin.transform(X)
        X = self.reciprocals.transform(X)
        X = self.interactions.transform(X)
        X = pd.concat([X, numbers], axis="columns")
        X = self.constant2.transform(X)
        X = self.selection2.transform(X)
        X = self.scaler2.transform(X)
        print("> Getting Predictions")
        y = self.lasso.predict(X)

        end = time.time()
        self.run_time(start, end)

        print("Model Monitoring:")
        start = time.time()

        print("> Computing Feature Drift")
        self.monitor(X, y)

        end = time.time()
        self.run_time(start, end)

        return self.labeler.inverse_transform(y)

    def refit(self, X, y):
        # add the new data to the model data
        self.X = pd.concat([self.X, X], axis="index").reset_index(drop=True)
        self.y = pd.concat([self.y, y], axis="index").reset_index(drop=True)

        # encode labels for the target
        y = self.y.copy()
        self.labeler = LabelEncoder()
        y.iloc[:, 0] = self.labeler.fit_transform(y.iloc[:, 0].tolist())
        y = y.astype(int)
                    
        print("Model Retraining:")
        start = time.time()

        # transform the new data
        print("> Transforming The Updated Data")
        X = self.names1.fit_transform(self.X)
        X = self.datetime.fit_transform(X)
        X = self.categorical.fit_transform(X)
        X = self.names2.fit_transform(X)
        X = self.impute.fit_transform(X)
        X = self.constant1.fit_transform(X)
        X = self.selection1.fit_transform(X, y)
        X = self.scaler1.fit_transform(X)
        numbers = self.numbers.fit_transform(X)
        X = self.bin.fit_transform(X)
        X = self.reciprocals.fit_transform(X)
        X = self.interactions.fit_transform(X)
        X = pd.concat([X, numbers], axis="columns")
        X = self.constant2.fit_transform(X)
        X = self.selection2.fit_transform(X, y)
        X = self.scaler2.fit_transform(X)

        print("> Training Lasso")
        self.lasso.fit(X, y)
        
        end = time.time()
        self.run_time(start, end)

        print("Model Indicators:")
        start = time.time()
        
        print("> Extracting Important Features")
        self.importance(X)

        end = time.time()
        self.run_time(start, end)
    
    def performance(self, X, y):
        # compute Accuracy and F1
        predictions = self.lasso.predict(X)
        y = y.iloc[:,0].to_numpy()
        self.bootstrap(y, predictions)
        df = pd.DataFrame({
            "Accuracy": self.accuracy,
            "F1": self.f1,
        })
        self.accuracy = np.mean(self.accuracy)
        self.f1 = np.mean(self.f1)

        # plot Accuracy and F1
        if self.plots:
            self.histogram(
                df,
                x="Accuracy",
                bins=20,
                title="Histogram For Accuracy",
                font_size=16,
            )
            self.histogram(
                df,
                x="F1",
                bins=20,
                title="Histogram For F1",
                font_size=16,
            )

        # compute control limits for errors
        error = (y != predictions) * 1
        df = self.p(error, 10)

        # plot the control limits for errors
        in_control = df.loc[(df["Proportion"] >= df["Proportion LCL"]) & (df["Proportion"] <= df["Proportion UCL"])].shape[0]
        in_control /= df.shape[0]
        in_control *= 100
        in_control = f"{round(in_control, 2)}%"
        if self.plots:
            self.histogram(
                df,
                x="Proportion",
                vlines=[df["Proportion LCL"][0], df["Proportion UCL"][0]],
                bins=10,
                title=f"Histogram For Errors, {in_control} In Control",
                font_size=16,
            )
        self.in_control = in_control

        # show the confusion matrix
        predictions = self.labeler.inverse_transform(predictions)
        y = self.labeler.inverse_transform(y)
        labels = np.unique(np.concatenate((predictions, y)))
        self.confusion = confusion_matrix(
            y_true=y,   # rows
            y_pred=predictions,  # columns
            labels=labels,
        )
        self.confusion = pd.DataFrame(
            self.confusion, 
            columns=labels, 
            index=labels,
        )

    def importance(self, X):
        # get the feature importance to determine indicators of the target
        coefficients = pd.DataFrame(self.lasso.coef_).T
        importance = coefficients.abs()
        importance = importance.mean(axis="columns")
        indicators = pd.DataFrame({
            "Indicator": X.columns,
            "Importance": importance,
        })
        indicators = indicators.sort_values(
            by="Importance", 
            ascending=False,
        ).reset_index(drop=True)
        indicators = indicators.loc[indicators["Importance"] > 0]

        # plot the feature importance
        if self.plots:
            self.bar_plot(
                indicators,
                x="Indicator",
                y="Importance",
                title="Feature Importance",
                font_size=16,
            )
        self.indicators = indicators

    def monitor(self, X, y):
        y_name = self.y.columns[0]
        X[y_name] = y  # new data

        # transform the raw data
        modelX = self.names1.transform(self.X)
        modelX = self.datetime.transform(modelX)
        modelX = self.categorical.transform(modelX)
        modelX = self.names2.transform(modelX)
        modelX = self.impute.transform(modelX)
        modelX = self.constant1.transform(modelX)
        modelX = self.selection1.transform(modelX)
        modelX = self.scaler1.transform(modelX)
        numbers = self.numbers.transform(modelX)
        modelX = self.bin.transform(modelX)
        modelX = self.reciprocals.transform(modelX)
        modelX = self.interactions.transform(modelX)
        modelX = pd.concat([modelX, numbers], axis="columns")
        modelX = self.constant2.transform(modelX)
        modelX = self.selection2.transform(modelX)
        modelX = self.scaler2.transform(modelX)
        modely = self.y.copy()
        modely.iloc[:, 0] = self.labeler.transform(modely.iloc[:, 0].tolist())
        modely = modely.astype(int)
        df = pd.concat([modelX, modely], axis="columns")  # data we trained on

        # see if the distribtuion of the new data is the same as the data we trained on
        pvalues = list()
        for column in df.columns:
            pvalues.append(kstest(
                df[column].tolist(),
                X[column].tolist(),
            ).pvalue)
        pvalues = pd.DataFrame({
            "Feature": df.columns,
            "pvalue": pvalues,
        })
        pvalues = pvalues.sort_values(
            by="pvalue", 
            ascending=False,
        ).reset_index(drop=True)

        # plot the pvalues
        if self.plots:
            self.bar_plot(
                pvalues,
                x="Feature",
                y="pvalue",
                title="Feature Drift, Drift Detected If pvalue < 0.05",
                font_size=16,
            )
        self.drift = pvalues

    def bootstrap(self, y_true, y_pred):
        df = pd.DataFrame({
            "Actual": y_true,
            "Predict": y_pred,
        })

        self.accuracy = list()
        self.f1 = list()
        np.random.seed(0)
        seeds = np.random.random_integers(low=0, high=1e6, size=1000)

        # randomly sample Accuracy and F1 scores
        for i in range(1000):
            sample = df.sample(frac=0.5, replace=True, random_state=seeds[i])
            self.accuracy.append(accuracy_score(
                y_true=sample["Actual"].tolist(),
                y_pred=sample["Predict"].tolist(),
            ))
            self.f1.append(f1_score(
                y_true=sample["Actual"].tolist(),
                y_pred=sample["Predict"].tolist(),
            ))

    def p(self, x: list, n: int):
        # group the binary values
        groups = np.repeat(
            a=np.arange(len(x)//n + 1) + 1, 
            repeats=n,
        )[:len(x)]

        data = pd.DataFrame({
            "Group": groups,
            "X": x,
        })

        data = data.groupby(["Group"]).agg({"X": ["mean", "count"]}).reset_index()
        data.columns = [" ".join(col).strip() for col in data.columns.values]

        # center line
        Nbar = data["X count"].mean()
        Pbar = (data["X mean"] * data["X count"]).sum() / data["X count"].sum()

        # control limits
        P_UCL = Pbar + 3 * np.sqrt(Pbar * (1 - Pbar) / Nbar)
        P_LCL = Pbar - 3 * np.sqrt(Pbar * (1 - Pbar) / Nbar)
        P_CL = Pbar

        # results
        df = data[["Group"]].copy()
        df["Proportion"] = data["X mean"]
        df["Proportion UCL"] = P_UCL
        df["Proportion LCL"] = P_LCL
        df["Proportion CL"] = P_CL

        return df

    def histogram(self, df, x, bins=20, vlines=None, title="Histogram", font_size=None):
        bin_size = (df[x].max() - df[x].min()) / bins
        fig = px.histogram(df, x=x, title=title)
        if vlines is not None:
            for line in vlines:
                fig.add_vline(x=line)
        fig.update_traces(xbins=dict( # bins used for histogram
                size=bin_size,
            ))
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}plots{path_sep}{title}.html")

    def bar_plot(self, df, x, y, color=None, title="Bar Plot", font_size=None):
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}plots{path_sep}{title}.html")

    def run_time(self, start, end):
        duration = end - start
        if duration < 60:
            duration = f"{round(duration, 2)} Seconds"
        elif duration < 3600:
            duration = f"{round(duration / 60, 2)} Minutes"
        else:
            duration = f"{round(duration / 3600, 2)} Hours"
        print(duration)

    def folder(self, name):
        if not os.path.isdir(name):
            os.mkdir(name)

    def dump(self):
        # save the machine learning pipeline and data
        # fit() or validate() has to be called for the pipeline and indicators to exist
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}labeler", "wb") as f:
            pickle.dump(self.labeler, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}names1", "wb") as f:
            pickle.dump(self.names1, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}datetime", "wb") as f:
            pickle.dump(self.datetime, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}categorical", "wb") as f:
            pickle.dump(self.categorical, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}names2", "wb") as f:
            pickle.dump(self.names2, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}impute", "wb") as f:
            pickle.dump(self.impute, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}constant1", "wb") as f:
            pickle.dump(self.constant1, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}scaler1", "wb") as f:
            pickle.dump(self.scaler1, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}selection1", "wb") as f:
            pickle.dump(self.selection1, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}numbers", "wb") as f:
            pickle.dump(self.numbers, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}bin", "wb") as f:
            pickle.dump(self.bin, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}reciprocals", "wb") as f:
            pickle.dump(self.reciprocals, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}interactions", "wb") as f:
            pickle.dump(self.interactions, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}constant2", "wb") as f:
            pickle.dump(self.constant2, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}selection2", "wb") as f:
            pickle.dump(self.selection2, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}scaler2", "wb") as f:
            pickle.dump(self.scaler2, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}lasso", "wb") as f:
            pickle.dump(self.lasso, f)
        self.X.to_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}X.csv", index=False)
        self.y.to_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}y.csv", index=False)
        self.indicators.to_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}indicators.csv", index=False)
        try:  # predict() has to be called for drift to exist
            self.drift.to_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}drift.csv", index=False)
        except:
            pass
        try:  # validate() has to be called for accuracy, f1, confusion, and in_control to exist
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}accuracy", "wb") as f:
                pickle.dump(self.accuracy, f)
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}f1", "wb") as f:
                pickle.dump(self.f1, f)
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}in_control", "wb") as f:
                pickle.dump(self.in_control, f)
            self.confusion.to_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}confusion.csv")
        except:
            pass

    def load(self):
        # load the machine learning pipeline and data
        # fit() or validate() had to have been called for the pipeline and indicators to exist
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}labeler", "rb") as f:
            self.labeler = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}names1", "rb") as f:
            self.names1 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}datetime", "rb") as f:
            self.datetime = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}categorical", "rb") as f:
            self.categorical = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}names2", "rb") as f:
            self.names2 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}impute", "rb") as f:
            self.impute = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}constant1", "rb") as f:
            self.constant1 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}scaler1", "rb") as f:
            self.scaler1 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}selection1", "rb") as f:
            self.selection1 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}numbers", "rb") as f:
            self.numbers = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}bin", "rb") as f:
            self.bin = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}reciprocals", "rb") as f:
            self.reciprocals = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}interactions", "rb") as f:
            self.interactions = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}constant2", "rb") as f:
            self.constant2 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}selection2", "rb") as f:
            self.selection2 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}scaler2", "rb") as f:
            self.scaler2 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}lasso", "rb") as f:
            self.lasso = pickle.load(f)
        self.X = pd.read_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}X.csv")
        self.y = pd.read_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}y.csv")
        self.indicators = pd.read_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}indicators.csv")
        try:  # predict() had to have been called for drift to exist
            self.drift = pd.read_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}drift.csv")
        except:
            pass
        try:  # validate() had to have been called for accuracy, f1, confusion, and in_control to exist
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}accuracy", "rb") as f:
                self.accuracy = pickle.load(f)
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}f1", "rb") as f:
                self.f1 = pickle.load(f)
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}in_control", "rb") as f:
                self.in_control = pickle.load(f)
            self.confusion = pd.read_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}confusion.csv")
        except:
            pass


class FeatureNames:
    def __init__(self, rename=True, verbose=True):
        self.rename = rename
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.rename:
            return self
        if self.verbose:
            print("> Renaming Features")

        self.columns = [re.sub(" ", "_", col) for col in X.columns]
        return self

    def transform(self, X, y=None):
        if not self.rename:
            return X
        
        X = X.copy()
        X.columns = self.columns
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class TimeFeatures:
    def __init__(self, time=True, verbose=True):
        self.time = time
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.time:
            return self
        if self.verbose:
            print("> Extracting Time Features")

        # convert any timestamp columns to datetime data type
        df = X.apply(
            lambda col: pd.to_datetime(col, errors="ignore")
            if col.dtypes == object 
            else col, 
            axis=0,
        )

        # check if any columns are timestamps
        self.features = [col for col in df.columns if is_datetime(df[col])]
        return self

    def transform(self, X, y=None):
        if not self.time:
            return X

        if len(self.features) == 0:
            return X
        else:
            X = X.copy()
            
            # convert any timestamp columns to datetime data type
            df = X[self.features].apply(
                lambda col: pd.to_datetime(col, errors="ignore")
                if col.dtypes == object 
                else col, 
                axis=0,
            )

            # extract timestamp features
            dt = pd.DataFrame()
            for col in self.features:
                # timestamp components
                dt[f"{col}_year"] = df[col].dt.isocalendar().year
                dt[f"{col}_quarter"] = df[col].dt.quarter
                dt[f"{col}_month"] = df[col].dt.month
                dt[f"{col}_week"] = df[col].dt.isocalendar().week
                dt[f"{col}_day_of_month"] = df[col].dt.day
                dt[f"{col}_day_of_week"] = df[col].dt.day_name()
                dt[f"{col}_hour"] = df[col].dt.hour
                dt[f"{col}_minute"] = df[col].dt.minute
                
                # economic data
                dates = df[col].dt.date
                start = min(dates)
                end = max(dates)
                fred = pdr.DataReader([
                    "NASDAQCOM", 
                    "UNRATE", 
                    "CPALTT01USM657N", 
                    "PPIACO",
                    "GDP",
                    "GDI",
                    "FEDFUNDS",
                ], "fred", start, end).reset_index()
                seq = pd.DataFrame({"DATE": pd.date_range(start=start, end=end)})
                fred = seq.merge(right=fred, how="left", on="DATE")
                fred = fred.ffill().bfill()  # fill in missing values with the last known value
                dt_fred = pd.DataFrame({"DATE": pd.to_datetime(dates)})
                dt_fred = dt_fred.merge(right=fred, how="left", on="DATE")
                dt[f"{col}_nasdaq"] = dt_fred["NASDAQCOM"]
                dt[f"{col}_unemployment"] = dt_fred["UNRATE"]
                dt[f"{col}_cpi"] = dt_fred["CPALTT01USM657N"]
                dt[f"{col}_ppi"] = dt_fred["PPIACO"]
                dt[f"{col}_gdp"] = dt_fred["GDP"]
                dt[f"{col}_gdi"] = dt_fred["GDI"]
                dt[f"{col}_federal_funds_rate"] = dt_fred["FEDFUNDS"]

            dt = pd.concat([X.drop(columns=self.features), dt], axis="columns")
            return dt

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class CategoricalFeatures:
    def __init__(self, binary=True, verbose=True):
        self.binary = binary
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.binary:
            return self
        if self.verbose:
            print("> Transforming Categorical Features")

        strings = X.select_dtypes(include="object").columns.tolist()
        df = X.copy().drop(columns=strings)
        numbers = [col for col in df.columns if len(df[col].unique()) <= 60 and (df[col] % 1 == 0).all()]
        self.categorical = strings + numbers
        if len(self.categorical) == 0:
            return self
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return self.encoder.fit(X[self.categorical].astype(str))

    def transform(self, X, y=None):
        if not self.binary:
            return X

        if len(self.categorical) == 0:
            return X
        continuous = X.copy().drop(columns=self.categorical)
        binary = self.encoder.transform(X[self.categorical].astype(str))
        binary = pd.DataFrame(binary, columns=self.encoder.get_feature_names_out())
        df = pd.concat([continuous, binary], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ImputeFeatures:
    def __init__(self, imputation=True, verbose=True):
        self.imputation = imputation
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.imputation:
            return self
        if self.verbose:
            print("> Filling In Missing Values")

        self.columns = X.columns
        self.imputer = KNNImputer()
        return self.imputer.fit(X)

    def transform(self, X, y=None):
        if not self.imputation:
            return X

        df = self.imputer.transform(X)
        df = pd.DataFrame(df, columns=self.columns)
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ConstantFeatures:
    def __init__(self, variance=True, verbose=True):
        self.variance = variance
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.variance:
            return self
        if self.verbose:
            print("> Removing Constant Features")

        self.selector = VarianceThreshold()
        return self.selector.fit(X)

    def transform(self, X, y=None):
        if not self.variance:
            return X

        df = self.selector.transform(X)
        df = pd.DataFrame(df, columns=self.selector.get_feature_names_out())
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ScaleFeatures:
    def __init__(self, scale=True, bounds=(0, 1), verbose=True):
        self.scale = scale
        self.bounds = bounds
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.scale:
            return self
        if self.verbose:
            print("> Scaling Features")
        
        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        if len(self.columns) == 0:
            return self
        self.scaler = MinMaxScaler(feature_range=self.bounds)
        return self.scaler.fit(X[self.columns])

    def transform(self, X, y=None):
        if not self.scale:
            return X

        if len(self.columns) == 0:
            return X
        df = self.scaler.transform(X[self.columns])
        df = pd.DataFrame(df, columns=self.columns)
        df = pd.concat([X.drop(columns=self.columns), df], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    
class AtwoodNumbers:
    def __init__(self, atwood=True, verbose=True):
        self.atwood = atwood
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.atwood:
            return self
        if self.verbose:
            print("> Computing Atwoood Numbers")

        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        return self

    def transform(self, X, y=None):
        if not self.atwood:
            return pd.DataFrame()

        if len(self.columns) < 2:
            return pd.DataFrame()
        numbers = list()
        pairs = list(combinations(self.columns, 2))
        for pair in pairs:
            numbers.append(pd.DataFrame({
                f"({pair[0]}-{pair[1]})/({pair[0]}+{pair[1]})": (X[pair[0]] - X[pair[1]]) / (X[pair[0]] + X[pair[1]]),
            }))
        df = pd.concat(numbers, axis="columns")
        df = df.fillna(0)
        df.replace(np.inf, 1e6, inplace=True)
        df.replace(-np.inf, -1e6, inplace=True)
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    

class BinFeatures:
    def __init__(self, binning=True, verbose=True):
        self.binning = binning
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.binning:
            return self
        if self.verbose:
            print("> Binning Features")
        
        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        if len(self.columns) == 0:
            return self
        self.binner = KBinsDiscretizer(n_bins=3, encode="onehot", strategy="uniform", subsample=None)
        return self.binner.fit(X[self.columns])

    def transform(self, X, y=None):
        if not self.binning:
            return X

        if len(self.columns) == 0:
            return X
        df = self.binner.transform(X[self.columns]).toarray()
        edges = self.binner.bin_edges_
        columns = list()
        for i, feature in enumerate(self.columns):
            bins = np.around(edges[i], 6)
            columns.append(f"{feature}({bins[0]}-{bins[1]})")
            columns.append(f"{feature}({bins[1]}-{bins[2]})")
            columns.append(f"{feature}({bins[2]}-{bins[3]})")
        df = pd.DataFrame(df, columns=columns)
        df = pd.concat([X, df], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
    
class Reciprocals:
    def __init__(self, reciprocal=True, verbose=True):
        self.reciprocal = reciprocal
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.reciprocal:
            return self
        if self.verbose:
            print("> Computing Reciprocals")

        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        return self

    def transform(self, X, y=None):
        if not self.reciprocal:
            return X

        if len(self.columns) == 0:
            return X
        df = 1 / X.copy()[self.columns]
        df.replace(np.inf, 1e6, inplace=True)
        df.replace(-np.inf, -1e6, inplace=True)
        df.columns = [f"1/{col}" for col in df.columns]
        df = pd.concat([X, df], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class Interactions:
    def __init__(self, interaction=True, verbose=True):
        self.interaction = interaction
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.interaction:
            return self
        if self.verbose:
            print("> Computing Interactions")

        self.interactions = PolynomialFeatures(
            degree=2, 
            interaction_only=True, 
            include_bias=False,
        )
        return self.interactions.fit(X)

    def transform(self, X, y=None):
        if not self.interaction:
            return X

        df = self.interactions.transform(X)
        columns = self.interactions.get_feature_names_out()
        columns = [re.sub(" ", "*", col) for col in columns]
        df = pd.DataFrame(df, columns=columns)
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class FeatureSelector:
    def __init__(self, selection=True, verbose=True):
        self.selection = selection
        self.verbose = verbose

    def fit(self, X, y):
        if not self.selection:
            return self
        if self.verbose:
            print("> Selecting Features")

        tree = XGBClassifier(
            booster="gbtree",
            n_estimators=25, 
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=0,
            n_jobs=-1,
        )
        tree.fit(X, y)

        # get the feature importance to determine indicators of the target
        importance = tree.feature_importances_
        indicators = pd.DataFrame({
            "Indicator": X.columns,
            "Importance": importance,
        })
        indicators = indicators.sort_values(
            by="Importance", 
            ascending=False,
        ).reset_index(drop=True)
        indicators = indicators.loc[indicators["Importance"] > 0]
        self.columns = indicators["Indicator"].tolist()

        return self

    def transform(self, X, y=None):
        if not self.selection:
            return X

        return X[self.columns]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
