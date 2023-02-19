# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:33:11 2023

@author: imyaash-admin
"""

import pandas as pd
import pickle as pkl
from tkinter import Tk, filedialog

def fraudDetector(df, featureSelectorPath, modelPath):
    featureSelector = pkl.load(open(featureSelectorPath, "rb"))
    clf = pkl.load(open(modelPath, "rb"))
    dfSelected = featureSelector.transform(df)
    df["PredictedClass"] = clf.predict(dfSelected)
    return df

if __name__ == '__main__':
    # Create a Tkinter GUI window to select files
    root = Tk()
    root.withdraw()
    dataPath = filedialog.askopenfilename(title="Select data file")
    featureSelectorPath = filedialog.askopenfilename(title="Select feature selector model file")
    clfPath = filedialog.askopenfilename(title="Select classification model file")
    outputPath = filedialog.asksaveasfilename(title="Save output file")

    # Load the input data
    df = pd.read_csv(dataPath)
    print("Data Loaded")
    print("Prediction Model Running")

    # Call the fraudDetector function
    dfPredicted = fraudDetector(df, featureSelectorPath, clfPath)
    print("Predictions Computed")
    print(dfPredicted[dfPredicted["PredictedClass"] == 1]["PredictedClass"].sum(), "Fraudulent Transactions Detected")
    print("Saving Output")

    # Save the output to a file
    dfPredicted.to_csv(outputPath, index=False)
    print("Outputfile Saved")
