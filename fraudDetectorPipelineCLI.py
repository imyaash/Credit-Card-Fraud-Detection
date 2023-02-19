# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 11:26:29 2023

@author: imyaash-admin
"""

import argparse
import pandas as pd
import pickle as pkl

def fraudDetector(df, featureSelectorPath, modelPath):
    featureSelector = pkl.load(open(featureSelectorPath, "rb"))
    clf = pkl.load(open(modelPath, "rb"))
    dfSelected = featureSelector.transform(df)
    df["PredictedClass"] = clf.predict(dfSelected)
    return df

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, help='Path to the input data file')
    parser.add_argument('--featureSelectorPath', type=str, help='Path to the feature selector model file')
    parser.add_argument('--clfPath', type=str, help='Path to the classification model file')
    parser.add_argument('--outputPath', type=str, help='Path to the output file')
    args = parser.parse_args()

    # Load the input data
    df = pd.read_csv(args.dataPath)
    print("Data Loaded")
    print("Prediction Model Running")

    # Call the fraudDetector function
    dfPredicted = fraudDetector(df, args.featureSelectorPath, args.clfPath)
    print("Predictions Computed")
    print(dfPredicted[dfPredicted["PredictedClass"] == 1]["PredictedClass"].sum(), "Fraudulent Transactions Detected")
    print("Saving Output")
    # Save the output to a file
    dfPredicted.to_csv(args.outputPath, index=False)
    print("Outputfile Saved")
