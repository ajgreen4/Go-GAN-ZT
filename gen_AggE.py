from __future__ import absolute_import, division, print_function, unicode_literals

# tensorflow
import tensorflow as tf

# standard python
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score
import math
import os
import pandas as pd

# plotting, especially for jupyter notebooks
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
# from IPython.display import Image
import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True # breaks for some endpoint labels
import seaborn as sns; sns.set()

# local routines
import useful_functions as uf

def calc_AggE(Ltoxicity, Lchemnames, Lgen_lab, Lfish, Lendpoints, Lconcentrations, verbose=0):
    """Determines active vs inactive compounds by calculating AggE from generated and true toxicity matrices
    and using impirically determined AggE thresholds (see Zhang et al. 2016).
    
    :parameter Ltoxicity: vector representing the real toxicity data.
            :type: np.array
    :parameter Lchemnames: array for unique chemical labels (CAS #).
            :type: np.array
    :parameter Lgen_lab: generator predicted toxicity matrices.
            :type: np.array
    :parameter Lfish: list of number of fish used to produce experimental data.
            :type: list of ints
    :parameter Lendpoints: which of the endpoints to used, from [0,1,2,3,4,5,...,14].
            :type: list of ints
    :parameter Lconcentrations: which of the concentrations to used, from [0,1,2,3,4,5].
            :type: list of ints
    :parameter verbose: Print info or not
    :type verbose: Boolean

    
    :returns: gen_activity_table, tox_activity_table
        :rtype: np.array
        gen_activity_table -- generatored list of active (1) and inactive (0) compounds.
        tox_activity_table -- impirical list of active (1) and inactive (0) compounds.
    """
    
    # import table of active compound in the validation set as determined by Zhang et al. 2016
    tox_active_AggE_file = '/home2/ajgreen4/Read-Across_w_GAN/DataFiles/tox_active_AggE.xlsx'
    tox_active_AggE_data = pd.read_excel(tox_active_AggE_file)
    tox_min = uf.truncate(float(tox_active_AggE_data.min(numeric_only=True)), 4)
    if verbose:
        print("Active threshold: ", tox_min)

    five_dpf = 0 # use only 5 dpf Lendpoints by setting to 4, 0 to use all
    rows = len(Lendpoints) - five_dpf 
    bio_state = rows + 1 # This number is used to calculate AggE see Zhang (2016) for details
    tox_Agge = gen_AggE = gen_activity_table = tox_activity_table = []
    for i in range(len(Ltoxicity)):
        chem = np.array(Lchemnames[i])
        genmat = []
        gen_cast = Lgen_lab[i,:]
        for j in range(five_dpf,len(Lendpoints)):
            start = len(Lconcentrations) * j
            end = start + len(Lconcentrations)
            if j == five_dpf:
                genmat = gen_cast[start:end]
            else:
                genmat = np.row_stack((genmat,gen_cast[start:end]))
        genmat = np.round((genmat[:,Lconcentrations] * Lfish[i,Lconcentrations]))
        gen_tsum = np.sum(genmat, axis=0)
        gen_NOAE = ((bio_state-gen_tsum)/bio_state)+(Lfish[i]-1)
#         AggE = (gen_tsum*0.1869)+0.2024
        AggE = (gen_tsum**-0.000002222)+(gen_tsum*0.1903)+0.07090
        
        active_state = (AggE >= tox_min).astype(int)
        AggE = np.append(chem,AggE)

        if i == 0:
            gen_AggE = AggE
            gen_activity_table = active_state
        else:
            gen_AggE = np.row_stack((gen_AggE,AggE))
            gen_activity_table = np.row_stack((gen_activity_table,active_state))
        if verbose:
            print("Chemical: ", chem)
            print(Lgen_lab[i,:])
            print("Predicted toxicity matrix:")
            print(genmat)
            print("Predicted tox.sum: ", gen_tsum)
            print("Predicted NOAE: ", gen_NOAE)
            print("Predicted AggE: ", AggE)

        tox_cast = Ltoxicity[i,:]
        toxmat = []
        for j in range(five_dpf,len(Lendpoints)):
            start = len(Lconcentrations) * j
            end = start + len(Lconcentrations)
            if j == five_dpf:
                toxmat = tox_cast[start:end]
            else:
                toxmat = np.row_stack((toxmat,tox_cast[start:end]))
        toxmat = np.round((toxmat[:,Lconcentrations] * Lfish[i,Lconcentrations]))
        tox_tsum = np.sum(toxmat, axis=0)
        tox_NOAE = ((bio_state-tox_tsum)/bio_state)+(Lfish[i]-1)
#         AggE = (tox_tsum*0.1869)+0.2024
        AggE = (tox_tsum**-0.000002222)+(tox_tsum*0.1903)+0.07090
        # Work around due to diffenrences between calculating AggE from sum vs indiv. matrices
        # WILL NEED TO BE REVISITED!!
        if (chem == '2164-17-2'):
            AggE = np.array(8.815420959)
        active_state = (AggE >= tox_min).astype(int)
        
        AggE = np.append(chem,AggE)
        if i == 0:
            tox_AggE = AggE
            tox_activity_table = active_state
        else:
            tox_AggE = np.row_stack((tox_AggE,AggE))
            tox_activity_table = np.row_stack((tox_activity_table,active_state))
        if verbose:
            print("Real toxicity matrix:")
            print("Fish: ", Lfish[i,Lconcentrations])
            print(Ltoxicity[i,:])
            print("Tox Matric: ", toxmat)
            print("Real tox.sum: ", tox_tsum)
            print("Real NOAE: ", tox_NOAE)
            print("Real AggE: ", AggE, " - Active State: ", active_state)
   
    return gen_activity_table, tox_activity_table, gen_AggE, tox_AggE


def calc_AggE_indv(Ltoxicity, Lchem_labels, Lchemnames, Lgen_lab, Lfish, Lendpoints, Lfactor, verbose=0):
    """Determines active vs inactive compounds by calculating AggE from generated and true toxicity matrices
    and using impirically determined AggE thresholds (see Zhang et al. 2016).
    
    :parameter Ltoxicity: vector representing the real toxicity data.
            :type: np.array
    :parameter Lchemnames: array for unique chemical labels (CAS #).
            :type: np.array
    :parameter Lchemnames: array of encoded chemical labels.
            :type: np.array
    :parameter Lgen_lab: generator predicted toxicity matrices.
            :type: np.array
    :parameter Lfish: list of number of fish used to produce experimental data.
            :type: list of ints
    :parameter Lendpoints: which of the endpoints to used, from [0,1,2,3,4,5,...,14].
            :type: list of ints
    :parameter Lfactor: array of expansion factors per chemical
            :type: np.array
    
    :returns: gen_activity_table, tox_activity_table
        :rtype: np.array
        gen_activity_table -- generatored list of active (1) and inactive (0) compounds.
        tox_activity_table -- impirical list of active (1) and inactive (0) compounds.
    """
    
    # import table of active compound in the validation set as determined by Zhang et al. 2016
    tox_active_AggE_file = '/home2/ajgreen4/Read-Across_w_GAN/DataFiles/tox_active_AggE.xlsx'
    tox_active_AggE_data = pd.read_excel(tox_active_AggE_file)
    tox_min = uf.truncate(float(tox_active_AggE_data.min(numeric_only=True)), 4)
    if verbose:
        print("Active threshold: ", tox_min)
    
    five_dpf = 0 # use only 5 dpf endpoints by setting to 4, 0 to use all
    rows = len(Lendpoints) - five_dpf 
    min_con = 10**-45 #deal with 0*log(0)
    bio_state = rows + 1 # This number is used to calculate AggE see Zhang (2016) for details
    
    # Calculate AggE real toxicity data
    tox_sum = np.sum(Ltoxicity, axis=1, keepdims=True)
    tox_NOAE = (bio_state-tox_sum)/bio_state
    tox_NOAE[tox_NOAE < 0] = 10**-45
    toxcast_AggE = (-1)*tox_sum*(1/bio_state)*np.log(1/bio_state)+(-1)*tox_NOAE*np.log(tox_NOAE+min_con)
#     toxcast_AggE = np.concatenate((tox_sum, tox_NOAE, toxcast_AggE), axis=1)
    
    i = 0
    for labels in Lchem_labels:
        labels = int(labels)
        results = np.where(Lfish == labels)
        try:
            chem_matricies =  toxcast_AggE[results[0]]
        except:
            print(toxcast_AggE)
        tox_tsum = np.sum(tox_sum[results[0]], axis=0, keepdims=True)
        tox_tNOAE = np.sum(tox_sum[results[0]], axis=0, keepdims=True)
        AggE = np.sum(chem_matricies, axis=0, keepdims=True)
        active_state = (AggE >= tox_min).astype(int)
        AggE = np.append(Lchemnames[labels],AggE)

        if i == 0:
            tox_AggE = AggE
            tox_activity_table = active_state
        else:
            tox_AggE = np.row_stack((tox_AggE,AggE))
            tox_activity_table = np.row_stack((tox_activity_table,active_state))
        i+=1
        if verbose:
            print("Real toxicity matrix:")
            print("Real tox.sum: ", tox_tsum)
            print("Real NOAE: ", tox_tNOAE)
            print("Real AggE: ", AggE, " - Active State: ", active_state, "\n")
            
    # Calculate AggE generated toxicity data
    # Duplicate generated toxicity by the number of fish in real toxicity data
    expanded_gen_lab = np.array(tf.repeat(Lgen_lab, repeats=Lfactor, axis=0))
    expanded_labels = np.array(tf.repeat(Lchem_labels, repeats=Lfactor, axis=0))
    
    tox_sum = np.sum(expanded_gen_lab, axis=1, keepdims=True)
    tox_NOAE = (bio_state-tox_sum)/bio_state
    tox_NOAE[tox_NOAE < 0] = 10**-45
    toxcast_AggE = (-1)*tox_sum*(1/bio_state)*np.log(1/bio_state)+(-1)*tox_NOAE*np.log(tox_NOAE+min_con)
#   toxcast_AggE = np.concatenate((tox_sum, tox_NOAE, toxcast_AggE), axis=1)
    
    i = 0
    for labels in Lchem_labels:
        labels = int(labels)
        results = np.where(expanded_labels == labels)
        chem_matricies =  toxcast_AggE[results[0]]
        gen_tsum = np.sum(tox_sum[results[0]], axis=0, keepdims=True)
        gen_tNOAE = np.sum(tox_sum[results[0]], axis=0, keepdims=True)
        AggE = np.sum(chem_matricies, axis=0, keepdims=True)
        active_state = (AggE >= tox_min).astype(int)
        AggE = np.append(Lchemnames[labels],AggE)

        if i == 0:
            gen_AggE = AggE
            gen_activity_table = active_state
        else:
            gen_AggE = np.row_stack((gen_AggE,AggE))
            gen_activity_table = np.row_stack((gen_activity_table,active_state))
        i+=1
        if verbose:
            print("Predicted toxicity matrix:")
            print("Predicted tox.sum: ", gen_tsum)
            print("Predicted NOAE: ", gen_tNOAE)
            print("Predicted AggE: ", AggE, " - Active State: ", active_state, "\n")
            
    return gen_activity_table, tox_activity_table, gen_AggE, tox_AggE

def display_conf_matrix(y_pred, y_true, Gmodelname=None, Display=False, Save_path=None):
    """Create confusion matrix and display model fit metrics.
    
    :parameter y_pred: predicted list of abnormal (1) and normal (0) values.
            :type: np.array
    :parameter y_true: impirical list of abnormal (1) and normal (0) values.
            :type: np.array
    :parameter Gmodelname: image output file name containing the type of model.
            :type: string
    :parameter Display: Display confusion matrix.
            :type: Boolean
    :parameter Save_path: Path were the confusion matrix should be saved to file.
                            if None don't save.
            :type: str
            
    :returns: NN_kappa_score, NN_auroc, SE, SP, PPV
    :rtype: int
        NN_kappa_score -- Kappa Statistic.
        NN_auroc -- Area under the Receiver Operating Curve.
        SE -- Sensitivity
        SP --  Specificity
        PPV -- Positive Predictive Value
    """
    
    y_pred = y_pred
    y_true = y_true
    conf_matrix = confusion_matrix(y_true, y_pred)
    tp = conf_matrix[1,1]
    tn = conf_matrix[0,0]
    conf_matrix[1,1] = tn
    conf_matrix[0,0] = tp
    # print(conf_matrix)
    NN_kappa_score = round(cohen_kappa_score(y_true, y_pred),3)
    NN_auroc = round(roc_auc_score(y_true, y_pred),4)
    

    mpl.style.use('seaborn')
    cm = pd.DataFrame(conf_matrix, 
                        index = ['Active', 'Inactive'],
                        columns = ['Active', 'Inactive'])    
    cm_rsum = np.sum(conf_matrix, axis=1, keepdims=True)
    cm_csum = np.sum(conf_matrix, axis=0, keepdims=True)
    cm_max = np.sum(conf_matrix)
    
    cm_perc = cm / cm_max.astype(float) * 100
    
    cm_csum = pd.DataFrame(cm_csum, 
                               index = ['Real Toxicity Data'],
                               columns = ['Active', 'Inactive'])
    cm_rsum = pd.DataFrame(cm_rsum, 
                               index = ['Active', 'Inactive'],
                               columns = ['Generated Toxicity Data'])

    cm = pd.concat([cm, cm_csum], axis=0)
    cm = pd.concat([cm, cm_rsum], axis=1, sort=False)
    cm.at['Real Toxicity Data','Generated Toxicity Data'] = cm.iloc[0, 0] + cm.iloc[1, 1]
    annot = np.empty_like(cm).astype(str)
    
    cmp = cm_perc.copy()
    cm_perc = np.zeros([cmp.shape[0]+1,cmp.shape[1]+1])
    cm_perc[0:2,0:2] = cmp  
    cm_perc = pd.DataFrame(cm_perc)

    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm.iloc[i, j]
            if i < 2 and j < 2:
                p = cm_perc.iloc[i, j]     
                if c <= 0:
                    annot[i, j] = '{}\n0.0%'.format(c)
                else:
                    annot[i, j] = '{}\n{:2.1f}%'.format(c,p)
                cm_perc.iloc[i, j] = 0
            elif i == 2 and j == 2:
                p = c / cm_max.astype(float) * 100
                annot[i, j] = 'ACC\n{:2.1f}%'.format(p)
                cm_perc.iloc[i, j] = p
            elif i == 2:
                p = cm.iloc[j,j] / c * 100
                if j == 0:
                    annot[i, j] = 'SE\n{:2.1f}%'.format(p)
                    SE = round(p, 1)
                    cm_perc.iloc[i, j] = p
                if j == 1:
                    annot[i, j] = 'SP\n{:2.1f}%'.format(p)
                    SP = round(p, 1)
                    cm_perc.iloc[i, j] = p
            elif j == 2:
                p = cm.iloc[i,i] / c * 100
                if i == 0:
                    annot[i, j] = 'PPV\n{:2.1f}%'.format(p)
                    PPV = round(p, 1)
                    cm_perc.iloc[i, j] = p
                if i == 1:
                    annot[i, j] = 'NPV\n{:2.1f}%'.format(p)
                    cm_perc.iloc[i, j] = p
    cm_perc.index.name = 'Generated Toxicity Data'
    cm_perc.columns.name = 'Real Toxicity Data'
    
    if Display:
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        cmap = sns.cubehelix_palette(light=0.85, dark=0.15, as_cmap=True)
#         res = sns.heatmap(cm_perc, annot=annot, vmin=0.0, vmax=cm_max, fmt='', ax=ax, cmap=cmap)        
        res = sns.heatmap(cm_perc, annot=annot, vmin=0.0, vmax=100, fmt='', ax=ax, cmap=cmap)
        plt.yticks([0.5,1.5], ['Active', 'Inactive'],va='center')
        plt.xticks([0.5,1.5], ['Active', 'Inactive'],va='center')

        if Gmodelname is not None:
            if Save_path is None:
                imageOut = '/home2/ajgreen4/Read-Across_w_GAN/imageOut/'
            else:
                imageOut = Save_path
            ## Is this a regression generator?
            if (Gmodelname.find('Go-ZT')!=-1):
                plt.title('Go-ZT Confusion Matrix')
                cm_file = imageOut+Gmodelname+'-confusion_matrix.png'
            elif (Gmodelname.find('GAN-ZT')!=-1):
                plt.title('GAN-ZT Confusion Matrix')
                cm_file = imageOut+Gmodelname+'-confusion_matrix.png' 
            else:
                plt.title(Gmodelname+' \nConfusion Matrix')
                cm_file = imageOut+Gmodelname+'-confusion_matrix.png'
        print('\n\nKappa: ',NN_kappa_score)       
        print('Auroc: ',NN_auroc)    
    if Save_path is not None:
        plt.savefig(cm_file, dpi=600, bbox_inches='tight')  
    plt.pause(0.5) 
    
    return NN_kappa_score, NN_auroc, SE, SP, PPV

def display_conf_matrix_simple(y_pred, y_true, Gmodelname=None, Display=False, Save_path=None):
    """Create simple confusion matrix and display model fit metrics.
    
    :parameter y_pred: predicted list of abnormal (1) and normal (0) values.
            :type: np.array
    :parameter y_true: impirical list of abnormal (1) and normal (0) values.
            :type: np.array
    :parameter Gmodelname: image output file name containing the type of model.
            :type: string
    :parameter Display: Display confusion matrix.
            :type: Boolean
    :parameter Save_path: Path were the confusion matrix should be saved to file.
                            if None don't save.
            :type: str
    """
    
    y_pred = y_pred
    y_true = y_true
    conf_matrix = confusion_matrix(y_true, y_pred)
    tp = conf_matrix[1,1]
    tn = conf_matrix[0,0]
    conf_matrix[1,1] = tn
    conf_matrix[0,0] = tp
    # print(conf_matrix)
    NN_kappa_score = round(cohen_kappa_score(y_true, y_pred),3)
    NN_auroc = round(roc_auc_score(y_true, y_pred),4)
    

    mpl.style.use('seaborn')
    cm = pd.DataFrame(conf_matrix, 
                        index = ['Active', 'Inactive'],
                        columns = ['Active', 'Inactive'])  

    cm_rsum = np.sum(conf_matrix, axis=1, keepdims=True)
    cm_csum = np.sum(conf_matrix, axis=0, keepdims=True)
    cm_max = np.sum(conf_matrix)
    
    cm_perc = cm / cm_max.astype(float) * 100
    
    annot = np.empty_like(cm).astype(str)
    
    cmp = cm_perc.copy()
    cm_perc = np.zeros([cmp.shape[0],cmp.shape[1]])
    cm_perc[0:2,0:2] = cmp  
    cm_perc = pd.DataFrame(cm_perc)

    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm.iloc[i, j]
            if i < 2 and j < 2:
                p = cm_perc.iloc[i, j]     
                if c <= 0:
                    annot[i, j] = '{}\n0.0%'.format(c)
                else:
                    annot[i, j] = '{}\n{:2.1f}%'.format(c,p)
                cm_perc.iloc[i, j] = p
                
    cm_perc.index.name = 'Generated Toxicity Data'
    cm_perc.columns.name = 'Real Toxicity Data'
    
    if Display:
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        cmap = sns.cubehelix_palette(light=0.85, dark=0.15, as_cmap=True)
#         res = sns.heatmap(cm_perc, annot=annot, vmin=0.0, vmax=cm_max, fmt='', ax=ax, cmap=cmap)        
        res = sns.heatmap(cm_perc, annot=annot, vmin=0.0, vmax=100, fmt='', ax=ax, cmap=cmap)
        plt.yticks([0.5,1.5], ['Active', 'Inactive'],va='center')
        plt.xticks([0.5,1.5], ['Active', 'Inactive'],va='center')

        if Gmodelname is not None:
            if Save_path is None:
                imageOut = '/home2/ajgreen4/Read-Across_w_GAN/imageOut/'
            else:
                imageOut = Save_path
            ## Is this a regression generator?
            if (Gmodelname.find('Go-ZT')!=-1):
                plt.title('Go-ZT Confusion Matrix')
                cm_file = imageOut+Gmodelname+'-simple_confusion_matrix.png'
            elif (Gmodelname.find('GAN-ZT')!=-1):
                plt.title('GAN-ZT Confusion Matrix')
                cm_file = imageOut+Gmodelname+'-simple_confusion_matrix.png' 
            else:
                plt.title(Gmodelname+' \nConfusion Matrix')
                cm_file = imageOut+Gmodelname+'-simple_confusion_matrix.png'
        print('\n\nKappa: ',NN_kappa_score)       
        print('Auroc: ',NN_auroc)    
    if Save_path is not None:
        plt.savefig(cm_file, dpi=600, bbox_inches='tight')  
    plt.pause(0.5) 

def display_random_conf_matrix(Lgen_activity_table, Ltox_activity_table, Display=0, Save_path=None):
    """Create confusion matrix and display model fit metrics.
    
    :parameter Lgen_activity_table: generatored list of active (1) and inactive (0) compounds.
            :type: np.array
    :parameter Ltox_activity_table: impirical list of active (1) and inactive (0) compounds.
            :type: np.array

    :parameter Display: Number to divide data by and display confusion matrix.
            :type: int
    :parameter Save: Save confusion matrix to file.
            :type: str
    """
    
    y_pred = Lgen_activity_table[:,-1]
    y_true = Ltox_activity_table[:,-1]
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix = np.round(conf_matrix/Display,0)  
    tp = conf_matrix[1,1]
    tn = conf_matrix[0,0]
    conf_matrix[1,1] = tn
    conf_matrix[0,0] = tp
    # print(conf_matrix)
    NN_kappa_score = round(cohen_kappa_score(y_true, y_pred),3)
    NN_auroc = round(roc_auc_score(y_true, y_pred),4)
    

    mpl.style.use('seaborn')
    cm = pd.DataFrame(conf_matrix, 
                        index = ['Active', 'Inactive'],
                        columns = ['Active', 'Inactive'])    
    cm_rsum = np.sum(conf_matrix, axis=1, keepdims=True)
    cm_csum = np.sum(conf_matrix, axis=0, keepdims=True)

    cm_perc = cm / cm_csum.astype(float) * 100

    cm_csum = pd.DataFrame(cm_csum, 
                               index = ['Real Toxicity Data'],
                               columns = ['Active', 'Inactive'])
    cm_rsum = pd.DataFrame(cm_rsum, 
                               index = ['Active', 'Inactive'],
                               columns = ['Generated Toxicity Data'])

    cm = pd.concat([cm, cm_csum], axis=0)
    cm = pd.concat([cm, cm_rsum], axis=1, sort=False)
    cm.at['Real Toxicity Data','Generated Toxicity Data'] = cm.iloc[0, 0] + cm.iloc[1, 1]
    annot = np.empty_like(cm).astype(str)
    cm_max = np.sum(conf_matrix)    

    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm.iloc[i, j]
            if i < 2 and j < 2:
                p = cm_perc.iloc[i, j]
                if c <= 0:
                    annot[i, j] = '{}\n0.0%'.format(c)
                else:
                    annot[i, j] = '{}\n{:2.1f}%'.format(c,p)
            elif i == 2 and j == 2:
                p = c / cm_max.astype(float) * 100
                annot[i, j] = 'ACC\n{:2.1f}%'.format(p)
            elif i == 2:
                p = cm.iloc[j,j] / c * 100
                if j == 0:
                    annot[i, j] = 'SE\n{:2.1f}%'.format(p)
                    SE = p
                if j == 1:
                    annot[i, j] = 'SP\n{:2.1f}%'.format(p)
                    SP = p
            elif j == 2:
                p = cm.iloc[i,i] / c * 100
                if i == 0:
                    annot[i, j] = 'PPV\n{:2.1f}%'.format(p)
                    PPV = p
                if i == 1:
                    annot[i, j] = 'FOR\n{:2.1f}%'.format(p)
    cm.index.name = 'Generated Toxicity Data'
    cm.columns.name = 'Real Toxicity Data'
    
    if Display:
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        cmap = sns.cubehelix_palette(light=0.85, dark=0.15, as_cmap=True)
        res = sns.heatmap(cm, annot=annot, vmin=0.0, vmax=cm_max, fmt='', ax=ax, cmap=cmap)
        plt.yticks([0.5,1.5], ['Active', 'Inactive'],va='center')
        plt.xticks([0.5,1.5], ['Active', 'Inactive'],va='center')
        plt.title('Random Confusion Matrix')
        plt.pause(0.5)    
        print('\n\nKappa: ',NN_kappa_score)
        print('Auroc: ',NN_auroc)   
        
    if Save_path is not None:
        imageOut = Save_path
        ## Is this a regression generator?
        cm_file = imageOut+'random_confusion_matrix.png'              
        plt.savefig(cm_file, dpi=600, bbox_inches='tight' )   
    
#     return NN_kappa_score, NN_auroc, SE, SP, PPV