from __future__ import absolute_import, division, print_function, unicode_literals

# standard python
import numpy as np
#import scipy
import pathlib
# pandas
import pandas as pd
#from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.preprocessing import LabelEncoder

def pessimisttoxform(toxmat, indv_matrix=False):
    """Convert toxicity endpoint matrix to a pessimistic alternative form.
    
    The original form encodes
    0 = not available since the fish died
    1 = effect (e.g. deformity) not observed
    2 = effect observed
    
    This form encodes
    0 = effect not observed or fish dead
    1 = effect observed
    
    Since mortality is also an endpoint, both forms include all the information.
    
    :parameter toxmat: the toxicity matrix in orignal form. It is modified in place
    :type toxmat: np.array
    :parameter indv_matrix: True toxmat is for a single fish
    :type toxmat: boolean
    
    rows are 
    ['MO24', 'DP24', 'SM24', 'NC24', 'MORT',
    'YSE_', 'AXIS', 'EYE_', 'SNOU', 'JAW_',
    'OTIC', 'PE__', 'BRAI', 'SOMI', 'PFIN',
    'CFIN', 'PIG_', 'CIRC', 'TRUN', 'SWIM',
    'NC__', 'TR__','Fish']
    
    """
    # last row 'Fish' is number of fish. Subtract it from all others, so [NA,no effect,effect]=[-1,0,1]
    toxmat[:-1] = toxmat[:-1] - toxmat[-1]
    # Row 0 'MO24' is number dead at 24 hours. 
    # Rows 1 to 3 'DP24', 'SM24', 'NC24', have a NA for each 'MO24'.
    # Add twice 'MO24' so NA becomes 1.
    toxmat[1:4] = toxmat[1:4] + 1 * toxmat[0]
    # Row 4 'MORT' has the (cumulative) number of dead at 5 days.
    # Add twice it to the remaining rows (except 'Fish')
    if indv_matrix:
        toxmat[5:] = toxmat[5:] + 1 * toxmat[4]
    else:
        toxmat[5:-1] = toxmat[5:-1] + 1 * toxmat[4]

    # Martin's original code with 1 = effect observed or fish dead
    # last row 'Fish' is number of fish. Subtract it from all others, so [NA,no effect,effect]=[-1,0,1]
#     toxmat[:-1] = toxmat[:-1] - toxmat[-1]
    # Row 0 'MO24' is number dead at 24 hours. 
    # Rows 1 to 3 'DP24', 'SM24', 'NC24', have a NA for each 'MO24'.
    # Add twice 'MO24' so NA becomes 1.
#     toxmat[1:3] = toxmat[1:3] + 2 * toxmat[0]
    # Row 4 'MORT' has the (cumulative) number of dead at 5 days.
    # Add twice it to the remaining rows (except 'Fish')
#     toxmat[5:-1] = toxmat[5:-1] + 2 * toxmat[4]


def xlsxfiletotoxicity(filename,concentration_indexes,endpoint_indexes,normalize=True,transform=False,decontrol=False,
                       indv_matrix=None, SET=0):
    """Convert an .xlsx file to a toxicity vector.
    
    :parameter filename: the name of the file
    :type filename: string
    

    :parameter normalize: If True, divides other entries by the number of Fish. 
        If False, enforses that the number of Fish is 32.
    :parameter transform: Load individual toxicity matrices and assign this int as their labels.
    :type transform: int
    :parameter transform: If True, changes encoding from [NA,no effect,effect]=[0,1,2] 
        to [no effect or NA (dead),effect]=[0,1]
    :type transform: Boolean
    :parameter indv_matrix: Sheetname to load and a class label.
    :type transform: list
    :parameter SET: Define row and column labels to use. 0: Standard concentration and endpoint labels
                    1: Super-endpoints 2: Individual matrix.
    :type transform: int
    :returns: np.array vector representing the toxicity.
    :rtype: np.array
    :returns: int representing the number of fish in that matrix or labels of individual matrices.
    :rtype: int
    
    """
    #Read the excel file and get a list of sheets.
    if indv_matrix is None:
        sheetname = 0
    else:
        sheetname = indv_matrix[0]
    df = pd.read_excel(filename, sheet_name=sheetname)

    if SET == 1:
        # validate Super-endpoints column names
        if False in set(df.columns == ['Unnamed: 0', '0.0064 uM', '0.064 uM', '0.64 uM', '6.4 uM','64 uM']):
            print("Column labels invalid for",filename,df.columns)
            raise ValueError
        # validate Super-endpoints row names
        if False in set(df[df.columns[0]] ==['MO24','DP24','SM24','NC24','MORT','SE2','AXIS','SE4','SE5',
                                            'TR__','PIG_','SE8','NC__','SE10', 'Fish',]):
            print("Row labels invalid for",filename,df[df.columns[0]])
            raise ValueError
    elif SET == 2:
        # validate indv Test set row names
        if False in set(df[df.columns[0]] == ['MO24', 'DP24', 'SM24', 'NC24', 'MORT', 'YSE_', 'AXIS', 
                                              'EYE_', 'SNOU', 'JAW_', 'OTIC', 'PE__', 'BRAI', 'SOMI', 
                                              'PFIN', 'CFIN', 'PIG_', 'CIRC', 'TRUN', 'SWIM', 'NC__', 
                                              'TR__']):
            print("Row labels invalid for",filename,df[df.columns[0]])
            raise ValueError
    elif SET == 3:
        # validate Test set row names
        if False in set(df[df.columns[0]] == ['MO24', 'DP24', 'SM24', 'NC24', 'MORT', 'YSE_', 'AXIS', 
                                              'EYE_', 'SNOU', 'JAW_', 'OTIC', 'PE__', 'BRAI', 'SOMI', 
                                              'PFIN', 'CFIN', 'PIG_', 'CIRC', 'TRUN', 'SWIM', 'NC__', 
                                              'TR__', 'Fish']):
            print("Row labels invalid for",filename,df[df.columns[0]])
            raise ValueError
    else:
        # validate column names
        if False in set(df.columns == ['Unnamed: 0', '0 uM', '0.0064 uM', 
                                           '0.064 uM', '0.64 uM', '6.4 uM','64 uM']):
            print("Column labels invalid for",filename,df.columns)
            raise ValueError
        # validate row names
        if False in set(df[df.columns[0]] == ['MO24', 'DP24', 'SM24', 'NC24', 'MORT', 'YSE_', 'AXIS', 
                                              'EYE_', 'SNOU', 'JAW_', 'OTIC', 'PE__', 'BRAI', 'SOMI', 
                                              'PFIN', 'CFIN', 'PIG_', 'CIRC', 'TRUN', 'SWIM', 'NC__', 
                                              'TR__', 'Fish']):
            print("Row labels invalid for",filename,df[df.columns[0]])
            raise ValueError
    # load the actual data part
    tmat = np.array(df[df.columns[1:]],dtype='float32')
    if indv_matrix is None:
        fish = tmat[-1,concentration_indexes]
    else:
        fish = int(indv_matrix[1])
    if transform:
        # transform encoding, in place
        pessimisttoxform(tmat)
#         print('Transformed:\n',tmat)
    if normalize:
        # normalize by last row and remove it.
        tmat = tmat[:-1]/tmat[-1]
    elif indv_matrix is None:
        # must be 32
        if np.all(tmat[-1] == 32):
            # remove last row
            tmat = tmat[:-1]
        else:
            print("Number of Fish not 32 in",filename,tmat[-1])
            tmat = tmat[:-1]
            #raise ValueError
    # take only selected concentrations
    out = tmat[:,concentration_indexes]
    if decontrol:
        # subtract out control data
        out = out - np.outer(tmat[:,0],np.ones(out.shape[1],dtype='float32'))
    # take only selected endpoints
    out = out[endpoint_indexes,:]
    # flatten
    out = out.flatten()
    return out, fish

def load_tmats(path,chemnames,concentration_indexes=None,endpoint_indexes=None,transform=False,decontrol=False, SET=0, verbose=1):
    """Import toxicity matrices from a directory.
    
    :parameter path: root path for this dataset
    :type path: string
    :parameter chemnames: list of the chemical names to get toxicity matrices for
    :type chemnames: list of strings
    :parameter concentration_indexes: which of the concentrations to use, from [0,1,2,3,4,5]. None uses all.
    :type concentration_indexes: list of ints
    :parameter endpoint_indexes: which of the endpoints to use, from [0,1,2,3,4,5,...,21]. None uses all.
    :type endpoint_indexes: list of ints
    :parameter transform: False leaves original encoding [NA,no effect,effect]=[0,1,2].
        True changes to encoding [no effect or NA (dead),effect]=[0,1]
    :type transform: Boolean
    :parameter decontrol: False leaves control the same as other concentrations.
        True subtracts the control and remove concentration 0.
    :type decontrol: Boolean
    :parameter SET: Define row and column labels to use. 0: Standard concentration and endpoint labels
                    1: Super-endpoints 2: Ignore concentration (column) labels
    :type transform: int
    :parameter verbose: Print info or not
    :type verbose: Boolean
    :returns: toxicity, Ntoxicity, endpoints, concentrations, fish_list
        toxicity -- master data matrix. Rows correspond to chemicals and columns to toxicity measurements.
        endpoints -- List of string labels for the endpoints selected by endpoint_indexes.
        concentrations -- List of string labels for the concentrations selected by concentration_indexes.
        fish_list -- list of int's for the number of fish in each matrix.
        Column i of toxicity corresponds to endpoints[i//len(concentrations)] concentrations[i%len(concentrations)].
    """
    ## Option for normalizing the toxicity matrices by dividing by the number of Fish
    normalize = True # divide by the number of fish
    # normalize = False # warns if not 32 fish. Probably only useful for debugging data handling.
    
    tpath = path+"Tox_matrices/"
    # default to using whole matrix
    if concentration_indexes is None:
        concentration_indexes = [0,1,2,3,4,5]
    if endpoint_indexes is None:
        endpoint_indexes = [i for i in range(22)]
    if decontrol:
        print("Subtracting out control data (concentration 0).")
        concentration_indexes = [i for i in concentration_indexes if i != 0]
    # column names, not including the column of row names
    cols = ['0 uM', '0.0064 uM', '0.064 uM', '0.64 uM', '6.4 uM','64 uM']
    # row names, not including the last row 'Fish'
    rows = ['MO24', 'DP24', 'SM24', 'NC24', 'MORT', 'YSE_', 'AXIS', 'EYE_', 'SNOU', 'JAW_', 'OTIC', 'PE__', 'BRAI', 'SOMI', 'PFIN', 'CFIN', 'PIG_', 'CIRC', 'TRUN', 'SWIM', 'NC__', 'TR__','Fish']
    # pull out the column and row names used
    concentrations = [cols[i] for i in concentration_indexes]
    endpoints = [rows[i] for i in endpoint_indexes]
    
        
    if transform:
        print("Transforming encoding to [no effect,effect or NA (dead)]=[0,1].")    
    tmats = []
    fish_list=[]
    for c in chemnames:
        [tempmat, tempfish] = xlsxfiletotoxicity(tpath+c+'.xlsx',concentration_indexes,endpoint_indexes,
                                                 normalize=normalize,transform=transform,decontrol=decontrol,SET=SET)
        tmats.append(tempmat)
        fish_list.append(tempfish)
    toxicity = np.array(tmats)
    fish_list = np.array(fish_list)
    Ntoxicity = toxicity.shape[1]
    if verbose:
        print("Number of chemicals=",len(chemnames))
        print("Using concentrations",concentrations)
        print("Using endpoints:",endpoints) 
        print("Toxicity vector length Ntoxicity=",Ntoxicity)
    
    return toxicity, endpoints, concentrations, fish_list

def load_indv_tmats(path,chemnames,endpoint_indexes=None,transform=False,encode=True, SET=0, verbose=1):
    """Import toxicity matrices from a directory.
    
    :parameter path: root path for this dataset
    :type path: string
    :parameter chemnames: list of the chemical names to get toxicity matrices for
    :type chemnames: list of strings
    :parameter endpoint_indexes: which of the endpoints to use, from [0,1,2,3,4,5,...,21]. None uses all.
    :type endpoint_indexes: list of ints
    :parameter transform: False leaves original encoding [NA,no effect,effect]=[0,1,2].
        True changes to encoding [no effect or NA (dead),effect]=[0,1]
    :type transform: Boolean
    :parameter encode: True encode CAS numbers as unique integers
    :type transform: Boolean
    :parameter SET: Define row and column labels to use. 0: Standard concentration and endpoint labels
                    1: Super-endpoints 2: Ignore concentration (column) labels
    :type transform: int
    :parameter verbose: Print info or not
    :type verbose: Boolean
    :returns: toxicity, Ntoxicity, endpoints, concentrations, fish_label
        toxicity -- master data matrix. Rows correspond to chemicals and columns to toxicity measurements.
        Ntoxicity -- number of columns in toxicity.
        endpoints -- List of string labels for the endpoints selected by endpoint_indexes.
        concentrations -- List of string labels for the concentrations selected by concentration_indexes.
        fish_label -- np.array of classs label of each matrix.
        Column i of toxicity corresponds to endpoints[i//len(concentrations)] concentrations[i%len(concentrations)].
    """
    
    ## Option for normalizing the toxicity matrices by dividing by the number of Fish
    normalize = False # divide by the number of fish
    
    tpath = path+"Tox_matrices/"
    # default to using whole matrix
    concentration_indexes = [0]
    if endpoint_indexes is None:
        endpoint_indexes = [i for i in range(22)]
    # column names, not including the column of row names
    cols = ['64 uM']
#     cols = ['0 uM', '0.0064 uM', '0.064 uM', '0.64 uM', '6.4 uM','64 uM']
    # row names, not including the last row 'Fish'
    rows = ['MO24', 'DP24', 'SM24', 'NC24', 'MORT', 'YSE_', 'AXIS', 'EYE_', 'SNOU', 'JAW_', 'OTIC', 'PE__', 'BRAI', 'SOMI', 'PFIN', 'CFIN', 'PIG_', 'CIRC', 'TRUN', 'SWIM', 'NC__', 'TR__','Fish']
    # pull out the column and row names used
    concentrations = [cols[i] for i in concentration_indexes]
    endpoints = [rows[i] for i in endpoint_indexes]
    
    chem_labels = np.arange(len(chemnames))
    chem_labels = np.reshape(chem_labels, (chem_labels.shape[0],1))
    
    if transform:
        print("Transforming encoding to [no effect,effect or NA (dead)]=[0,1].")    
    tmats = []
    fish_label=[]
    for i in range(len(chemnames)):
        filename = tpath+chemnames[i]+'.xlsx'
        #Read the excel file and get a list of sheets.
        xls = pd.ExcelFile(filename)
        for sheet_name in xls.sheet_names:
            [tempmat, tempfish] = xlsxfiletotoxicity(filename,concentration_indexes,endpoint_indexes,
                                                     normalize=normalize,transform=transform, 
                                                     indv_matrix=[sheet_name,chem_labels[i]], SET=SET)
            tmats.append(tempmat)
            fish_label.append(tempfish)

    toxicity = np.array(tmats)
    fish_label = np.array(fish_label)
    fish_label = np.reshape(fish_label, (fish_label.shape[0],1))
    Ntoxicity = toxicity.shape[1]
    if verbose:
        print("Number of chemicals=",len(chemnames))
        print("Using concentrations",concentrations)
        print("Using endpoints:",endpoints) 
        print("Toxicity vector length Ntoxicity=",Ntoxicity)
    
    return toxicity, endpoints, concentrations, fish_label

def load_tmats_SE(path,chemnames,concentration_indexes,endpoint_indexes, transform=False,):
    """Import super endpoint toxicity matrices from a directory.
    
    :parameter path: root path for this dataset
    :type path: string
    :parameter chemnames: list of the chemical names to get toxicity matrices for
    :type chemnames: list of strings
    :parameter concentration_indexes: which of the concentrations to use, from [0,1,2,3,4,5]. None uses all.
    :type concentration_indexes: list of ints
    :parameter endpoint_indexes: which of the endpoints to use, from [0,1,2,3,4,5,...,14]. None uses all.
    :type endpoint_indexes: list of ints
    :parameter transform: False leaves original encoding [NA,no effect,effect]=[0,1,2].
        True changes to encoding [no effect,effect or NA (dead)]=[0,1]
    :type transform: Boolean
    :returns: toxicity, Ntoxicity, endpoints, concentrations
        toxicity -- master data matrix. Rows correspond to chemicals and columns to toxicity measurements.
        endpoints -- List of string labels for the endpoints selected by endpoint_indexes.
        concentrations -- List of string labels for the concentrations selected by concentration_indexes.
        Column i of toxicity corresponds to endpoints[i//len(concentrations)] concentrations[i%len(concentrations)].
    """    
    ## Option for normalizing the toxicity matrices by dividing by the number of Fish
    normalize = True # divide by the number of fish
    # normalize = False # warns if not 32 fish. Probably only useful for debugging data handling.

    tpath = path+"Tox_matrices/"
    # column names, not including the column of row names
    concentrations = ['0.0064 uM', '0.064 uM', '0.64 uM', '6.4 uM','64 uM']
    cols = [concentrations[i] for i in concentration_indexes]
    # row names, not including the last row 'Fish'
    endpoints = ['MO24','DP24','SM24','NC24','MORT','SE2','AXIS','SE4','SE5','TR__','PIG_','SE8','NC__','SE10']
    
    print("Using concentrations: ",cols)
#     print("Using endpoints:",[rows[i] for i in endpoints])     
    if transform:
        print("Transforming encoding to [no effect,effect or NA (dead)]=[0,1].")    
    tmats = []
    for c in chemnames:
        [tempmat, tempfish] = xlsxfiletotoxicity(tpath+c+'.xlsx',concentration_indexes,endpoint_indexes,
                                                 normalize=normalize,transform=transform, SE=1)
        tmats.append(tempmat)
        fish_list.append(tempfish)
    toxicity = np.array(tmats)
    Ntoxicity = toxicity.shape[1]
    print("Toxicity vector length Ntoxicity=",Ntoxicity)
    
    return toxicity, endpoints, cols

def toxicityxlsxtopandasscores(filename,meanother=True, verbose=1):
    """ Convert xlsx file of toxicity to scores in pandas.
    
    Always:
    * Changes encoding from [NA,no effect,effect]=[0,1,2] to [no effect,effect or NA (dead)]=[0,1]
    * Divides other entries by the number of Fish and removes that row.
    * Takes the max of the first 3 concentrations as the control c
        and the max of the last 3 as the treated x, 
        then computes (x-c)_+ / (1-c) and returns as a single column.
        Warns if control value large ( >0.5).
    
    :parameter filename: the name of the file
    :type filename: string
    :parameter meanother: If True, takes the mean of the non-mortality 24 hour endpoints to make a new endpoint 'OT24'
        and takes the mean of the non-mortality 5 day endpoints to make a new endpoint 'OTHR'.
    :type meanother: Boolean
    :returns: A column with either 22 or 4 (meanother=True) rows.
    :rtype: pandas frame
    :parameter verbose: Print info or not
    :type verbose: Boolean
    
    """

    # read in
    df = pd.read_excel(filename,index_col=0)
    # validate column names
    if False in set(df.columns == ['0 uM', '0.0064 uM', '0.064 uM', '0.64 uM', '6.4 uM','64 uM']):
        print("Column labels invalid for",filename,df.columns)
        raise ValueError
    # validate row names
    if False in set(df.index == ['MO24', 'DP24', 'SM24', 'NC24', 'MORT', 'YSE_', 'AXIS', 'EYE_', 'SNOU', 'JAW_', 'OTIC', 'PE__', 'BRAI', 'SOMI', 'PFIN', 'CFIN', 'PIG_', 'CIRC', 'TRUN', 'SWIM', 'NC__', 'TR__', 'Fish']):
        print("Row labels invalid for",filename,df[df.columns[0]])
        raise ValueError
    # static labels
    notFish = ['MO24', 'DP24', 'SM24', 'NC24', 'MORT', 'YSE_', 'AXIS', 'EYE_', 'SNOU', 'JAW_', 'OTIC', 'PE__', 'BRAI', 'SOMI', 'PFIN', 'CFIN', 'PIG_', 'CIRC', 'TRUN', 'SWIM', 'NC__', 'TR__']
    end24 = ['DP24', 'SM24', 'NC24']
    end5d = ['YSE_', 'AXIS', 'EYE_', 'SNOU', 'JAW_', 'OTIC', 'PE__', 'BRAI', 'SOMI', 'PFIN', 'CFIN', 'PIG_', 'CIRC', 'TRUN', 'SWIM', 'NC__', 'TR__'] 
    ##  transform:
    # shift [NA (dead), no effect, effect] from [0,1,2] to [-1,0,1]
    df.loc[notFish] -= df.loc['Fish']
    # add dead fish twice to rows other than mortality so [no effect,effect or dead] = [0,1]
    df.loc[end24] += 2* df.loc['MO24']
    df.loc[end5d] += 2*df.loc['MORT']
    ## normalize:
    df = df.loc[notFish] / df.loc['Fish']
    if meanother:
        df.loc['OT24'] = df.loc[end24].mean()
        df.loc['OTHR'] = df.loc[end5d].mean()
        df = df.loc[['MO24','OT24','MORT','OTHR']]
    # control as max of first 3 columns
    control = df[['0 uM','0.0064 uM','0.064 uM']].max(axis=1)
    mc = control.max()
    if mc > 0.5 and verbose:
        print("Warning: max control proportion",mc,"in",filename)
    # treat as max of last 3 columns
    treat = df[['0.64 uM','6.4 uM','64 uM']].max(axis=1)
    # take positive part of the difference
    dif = (treat - control).clip(lower=0)
    # divide by (1-control) and set as output
    df = dif/(1-control)
    # detect NA due to division by 0 and set to 0. corresponds to mc==1
    df.fillna(value=0.,inplace=True)
    return df

def load_tscores(path,chemnames,meanother=False,verbose=1):
    """Import toxicity matrices from a directory and return a matrix of scores.
    
    See toxicityxlsxtopandasscores for description of what the scores mean.
    
    :parameter path: root path for this dataset
    :type path: string
    :parameter chemnames: list of the chemical names to get toxicity matrices for
    :type chemnames: list of strings
    :parameter verbose: Print info or not
    :type verbose: Boolean
    :parameter meanother: If True, takes the mean of the non-mortality 24 hour endpoints to make a new endpoint 'OT24'
        and takes the mean of the non-mortality 5 day endpoints to make a new endpoint 'OTHR'.
    :type meanother: Boolean
    :returns: toxicity, Ntoxicity, endpoints
        toxicity -- master data matrix. Rows correspond to chemicals and columns to toxicity measurements.
        Ntoxicity -- number of columns in toxicity.
        endpoints -- List of string labels for the endpoints.
    """
    tpath = path+"Tox_matrices/"
    # default to using whole matrix
    pdscores = []
    for c in chemnames:
        pdscores.append(toxicityxlsxtopandasscores(tpath+c+'.xlsx',meanother=meanother, verbose=verbose))
    # row labels
    endpoints = list(pdscores[0].index)
    # count of them
    Ntoxicity = len(endpoints)   
    # convert to np. dtype='float32' is for tensorflow
    toxicity = np.array(pdscores,dtype='float32')
    if verbose:
        print("Number of chemicals=",len(chemnames))
        print("Using endpoints:",endpoints) 
        print("Toxicity vector length Ntoxicity=",Ntoxicity)
    
    return toxicity, Ntoxicity, endpoints

########################################
if __name__ == "__main__":
    if 0:
        # manual test of pessimisttoxform
        # read in one
        df = pd.read_excel('../DataFiles/Tox21_compounds_firstround/'+"Tox_matrices/"+'131-17-9'+'.xlsx')
        print(df)
        toxmat = np.array(df[df.columns[1:]],dtype='float32')
        print(toxmat)
        # transform, in place
        pessimisttoxform(toxmat)
        print(toxmat) 

    if 0: 
        # manual test of xlsxfiletotoxicity
        f = '../DataFiles/Tox21_compounds_firstround/'+"Tox_matrices/"+'131-17-9'+'.xlsx'
        if 0:
            tm = xlsxfiletotoxicity(f,concentration_indexes=[i for i in range(6)],endpoint_indexes=[i for i in range(22)],transform=True)
            #tm = xlsxfiletotoxicity(f,concentration_indexes=[-1],endpoint_indexes=[0,4],transform=True)
            print(tm)
        if 1: # testing decontrol
            f = '../DataFiles/Tox21_training_compounds/'+"Tox_matrices/"+'6915-15-7'+'.xlsx'
            tm = xlsxfiletotoxicity(f,concentration_indexes=[i for i in range(2)],endpoint_indexes=[i for i in range(4)],transform=True,decontrol=False)
            print(tm)
            tm = xlsxfiletotoxicity(f,concentration_indexes=[i for i in range(2)],endpoint_indexes=[i for i in range(4)],transform=True,decontrol=True)
            print(tm)
    if 0:
        # manual test of load_tmats
        path = '../DataFiles/Tox21_compounds_firstround/'
        chemnames = ['56-72-4','57-74-9']
        [toxicity, Ntoxicity, endpoints, concentrations] = load_tmats(path,chemnames,concentration_indexes=[0,3],endpoint_indexes=[0,3,7],transform=True,decontrol=True)
    if 0:
        # manual test of toxicityxlsxtopandasscores
        filename = '../DataFiles/Tox21_training_compounds/'+"Tox_matrices/"+'56-72-4'+'.xlsx'
        tm = toxicityxlsxtopandasscores(filename,meanother=True)
        print(tm)
    if 1:
        #manual test ot load_tscores
        path = '../DataFiles/Tox21_compounds_firstround/'
        chemnames = ['56-72-4','57-74-9']
        toxicity, Ntoxicity, endpoints = load_tscores(path,chemnames,meanother=True,verbose=1)
        print(toxicity)
