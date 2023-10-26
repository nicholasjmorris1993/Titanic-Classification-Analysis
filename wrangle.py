def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import time
import numpy as np
import pandas as pd


def prepare(df):
    data = Prepare(df)
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

    return data.df
    
    
class Prepare:
    def __init__(self, df):
        self.df = df  # dataset

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

    def run_time(self, start, end):
        duration = end - start
        if duration < 60:
            duration = f"{round(duration, 2)} Seconds"
        elif duration < 3600:
            duration = f"{round(duration / 60, 2)} Minutes"
        else:
            duration = f"{round(duration / 3600, 2)} Hours"
        print(duration)
