def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import re
import time
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.impute import KNNImputer
import scipy.cluster.hierarchy as sch
import plotly.express as px
from plotly.offline import plot

if os.name == "nt":
    path_sep = "\\"
else:
    path_sep = "/"


def prepare(df, name="Data Preparation", path=None, plots=True):
    data = Prepare(df, name, path, plots)
    print("Data Wrangling:")
    start = time.time()
    data.drop()
    data.titles()
    data.ticket()
    data.missing_values()
    data.cabin()
    data.shuffle()
    end = time.time()
    data.run_time(start, end)

    if plots:
        print("Plotting:")
        start = time.time()
        data.impute()
        data.correlations()
        data.scatter_plots()
        data.histograms()
        data.bar_plots()
        data.pairwise_bar_plots()
        data.boxplots()
        end = time.time()
        data.run_time(start, end)

    return data.df
    
    
class Prepare:
    def __init__(
        self, 
        df,
        name="Data Preparation", 
        path=None,
        plots=True,
    ):
        self.df = df  # dataset
        self.name = name  # name of the analysis
        self.path = path  # the path where results will be exported
        self.plots = plots  # should we plot the analysis?
        
        if self.path is None:
            self.path = os.getcwd()

        # create folders for output files
        if self.plots:
            self.folder(f"{self.path}{path_sep}{self.name}")
    
    def drop(self):
        print("> Removing Unnecessary Columns")
        self.df = self.df.drop(columns="PassengerId")
    
    def titles(self):
        print("> Getting Titles From Name")
        title = list()
        for name in self.df["Name"]:
            name = name.split(" ")
            for item in name:
                if "." in item:
                    title.append(item)
                    break
        self.df["Title"] = title
        self.df = self.df.drop(columns="Name")

    def ticket(self):
        print("> Transforming Ticket")
        # get the first character from Ticket
        ticket = list()
        for item in self.df["Ticket"]:
            ticket.append(item[0])
        self.df["Ticket"] = ticket

    def missing_values(self):
        print("> Replacing Missing Values")
        # replace missing values in Cabin and Embarked with None
        self.df["Cabin"] = self.df["Cabin"].fillna("None")
        self.df["Embarked"] = self.df["Embarked"].fillna("None")

    def cabin(self):
        print("> Transforming Cabin")
        # get the first character from Cabin
        cabin = list()
        for item in self.df["Cabin"]:
            cabin.append(item[0])
        self.df["Cabin"] = cabin
    
    def shuffle(self):
        print("> Shuffling The Data")
        self.df = self.df.sample(frac=1, random_state=0).reset_index(drop=True)

    def impute(self):
        # separate the numbers from the strings
        self.data = self.df.copy()
        numbers = self.data[["Survived", "Pclass", "Age", "SibSp", "Parch"]]
        strings = self.data.drop(columns=["Survived", "Pclass", "Age", "SibSp", "Parch"])
        
        # impute missing values
        columns = numbers.columns
        imputer = KNNImputer()
        numbers = imputer.fit_transform(numbers)
        numbers = pd.DataFrame(numbers, columns=columns)
        
        self.data = pd.concat([numbers, strings], axis="columns")

    def correlations(self):
        if self.plots:
            print("> Plotting Correlations")
            self.correlation_plot(
                df=self.data.drop(columns=["Sex", "Ticket", "Cabin", "Embarked", "Title"]), 
                title="Correlation Heatmap",
                font_size=16,
            )

    def scatter_plots(self):
        if self.plots:
            pairs = list(combinations(["Age", "Fare"], 2))
            for pair in pairs:
                print(f"> {pair[0]} vs. {pair[1]}")
                self.scatter_plot(
                    df=self.data,
                    x=pair[0],
                    y=pair[1],
                    color=None,
                    title=f"{pair[0]} vs. {pair[1]}",
                    font_size=16,
                )

    def histograms(self):
        if self.plots:
            for col in ["Age", "Fare"]:
                print(f"> Plotting {col}")
                self.histogram(
                    df=self.data,
                    x=col,
                    bins=20,
                    title=col,
                    font_size=16,
                )
    
    def bar_plots(self):
        if self.plots:
            for col in [
                "Survived", 
                "Pclass", 
                "Sex", 
                "SibSp", 
                "Parch", 
                "Ticket", 
                "Cabin", 
                "Embarked", 
                "Title",
            ]:
                print(f"> Plotting {col}")
                proportion = self.data[col].value_counts(normalize=True).reset_index()
                proportion.columns = ["Label", "Proportion"]
                proportion = proportion.sort_values(by="Proportion", ascending=False).reset_index(drop=True)
                self.bar_plot(
                    df=proportion,
                    x="Proportion",
                    y="Label",
                    title=col,
                    font_size=16,
                )

    def pairwise_bar_plots(self):
        if self.plots:
            pairs = list(combinations([
                "Survived", 
                "Pclass", 
                "Sex", 
                "SibSp", 
                "Parch", 
                "Ticket", 
                "Cabin", 
                "Embarked", 
                "Title",
            ], 2))
            for pair in pairs:
                print(f"> {pair[0]} vs. {pair[1]}")
                data = pd.DataFrame()
                data[f"{pair[0]}, {pair[1]}"] = self.data[pair[0]].astype(str) + ", " + self.data[pair[1]].astype(str)
                proportion = data[f"{pair[0]}, {pair[1]}"].value_counts(normalize=True).reset_index()
                proportion.columns = [f"{pair[0]}, {pair[1]}", "Proportion"]
                proportion = proportion.sort_values(by="Proportion", ascending=False).reset_index(drop=True)
                self.bar_plot(
                    df=proportion,
                    x="Proportion",
                    y=f"{pair[0]}, {pair[1]}",
                    title=f"{pair[0]} vs. {pair[1]}",
                    font_size=16,
                )

    def boxplots(self):
        if self.plots:
            pairs = list()
            for number in ["Age", "Fare"]:
                for string in [
                    "Survived", 
                    "Pclass", 
                    "Sex", 
                    "SibSp", 
                    "Parch", 
                    "Ticket", 
                    "Cabin", 
                    "Embarked", 
                    "Title",
                ]:
                    pairs.append((number, string))
            
            for pair in pairs:
                print(f"> {pair[0]} vs. {pair[1]}")
                # sort the data by the group average
                data = self.data.copy()
                df = data.groupby(pair[1]).agg({pair[0]: "mean"}).reset_index()
                df = df.sort_values(by=pair[0]).reset_index(drop=True).reset_index()
                df = df.drop(columns=pair[0])
                data = data.merge(right=df, how="left", on=pair[1])
                data = data.sort_values(by="index").reset_index(drop=True)
                data[pair[1]] = data[pair[1]].astype(str)
                self.box_plot(
                    df=data, 
                    x=pair[0], 
                    y=pair[1],
                    title=f"{pair[0]} vs. {pair[1]}",
                    font_size=16,
                )

    def correlation_plot(self, df, title="Correlation Heatmap", font_size=None):
        df = df.copy()
        correlation = df.corr()

        # group columns together with hierarchical clustering
        X = correlation.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method="ward")
        ind = sch.fcluster(L, 0.5*d.max(), "distance")
        columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
        df = df.reindex(columns, axis=1)
        
        # compute the correlation matrix for the received dataframe
        correlation = df.corr()

        # plot the correlation matrix
        fig = px.imshow(correlation, title=title, range_color=(-1, 1))
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def scatter_plot(self, df, x, y, color=None, title="Scatter Plot", font_size=None):
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

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
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def bar_plot(self, df, x, y, color=None, title="Bar Plot", font_size=None):
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def box_plot(self, df, x, y, color=None, title="Box Plot", font_size=None):
        fig = px.box(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

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

