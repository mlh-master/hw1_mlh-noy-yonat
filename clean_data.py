# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
        # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    tmp={}
    tmp= CTG_features.drop(axis = 'columns', labels = [extra_feature]) #remove DR column
    tmp=tmp.apply(lambda x: pd.to_numeric(x, errors='coerce')) #remove non numeric values 
    Dic = tmp.to_dict('series') #change to dict
    c_ctg={}
    for i in Dic:
       c_ctg[i]=Dic[i].dropna() #remove 'nan' values 
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):

    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        
    tmpCTG = CTG_features.drop(axis ='columns',labels = [extra_feature]) #drop 'RD' col
    #Iter the columns and replace nan  in random value:
    for column in tmpCTG.columns:
       rand_values=np.random.choice(tmpCTG[column])
       c_cdf[column]=pd.to_numeric(tmpCTG[column],errors='coerce').fillna(rand_values)

    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    d_summary = {}
    
    for feature in c_feat:
        d_summary[feature] = {}
        maximum = np.amax(c_feat.loc[:,feature].values)
        minn = np.amin(c_feat.loc[:,feature].values)
        Q1 = np.percentile(c_feat.loc[:,feature],25)
        Q3 = np.percentile(c_feat.loc[:,feature],75)
        median = np.median(c_feat.loc[:,feature].values)
        maximum = np.amax(c_feat.loc[:,feature].values)
    
        d_summary[feature] = {"min": minn, "Q1": Q1, "median": median,"Q3": Q3, "Max": maximum}
    # -------------------------------------------------------------------------
    return d_summary

def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    IQR = {}
    Q1 = {}
    Q3 = {}
    for feature in d_summary:
        IQR[feature] = d_summary[feature]['Q3'] - d_summary[feature]['Q1']
        Q1 = d_summary[feature]['Q1']
        Q3 = d_summary[feature]['Q3']
        right_lim = (Q1 + 1.5 * IQR[feature])
        left_lim = (Q3 - 1.5 * IQR[feature])
        t = c_feat[feature]
        c_no_outlier[feature] = t[(t <= right_lim) & (t >= left_lim)]
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)



def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    temp_phys=c_cdf.loc[:,feature].to_numpy() 
    filt_feature=temp_phys[temp_phys<thresh]
    # -------------------------------------------------------------------------
    return filt_feature

def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    # Parameters:
    nsd_ans ={}
    chosenFeatures = [x, y]
    b = 100

    # run over all the features:
    for feature in CTG_features.columns:
        if mode == 'standard':
            # define parameters:
            mu = CTG_features[feature].mean()
            sigma = CTG_features[feature].std()
            # calculate Z:
            CTGstandard = (CTG_features[feature] - mu) / sigma
            nsd_ans[feature] = CTGstandard
        elif mode == 'MinMax':
            # define parameters:
            minCTG = min(CTG_features[feature])
            maxCTG = max(CTG_features[feature])
            # calculate x norm:
            CTGnorm = (CTG_features[feature] - minCTG) / (maxCTG - minCTG)
            nsd_ans[feature] = CTGnorm
        elif mode == 'mean':
            # define parameters:
            minCTG = min(CTG_features[feature])
            maxCTG = max(CTG_features[feature])
            meanCTG = CTG_features[feature].mean()
            # calculate x mean norm:
            CTGnormmean = (CTG_features[feature] - meanCTG) / (maxCTG - minCTG)
            nsd_ans[feature] = CTGnormmean
        else:
            nsd_ans[feature] = CTG_features[feature]

    # plots for the chosen Features:
    if flag == True:
        chosenValues = [nsd_ans[chosenFeatures[0]], nsd_ans[chosenFeatures[1]]]
        plotVaules = pd.DataFrame(chosenValues).transpose()
        plotVaules.plot.hist(bins=b)

    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_ans)

