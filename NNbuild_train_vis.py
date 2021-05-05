from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import multi_gpu_model

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import pathlib
import pandas as pd
#from pandas import ExcelWriter
from pandas import ExcelFile


# Losses and optimizers
# losses
Gloss_function = tf.keras.losses.MeanSquaredError()
Dloss_function = tf.keras.losses.MeanSquaredError()

# optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

# generic dense NN
def multiDense(Nin,Nout,Nhidden,NDrop=0,Ndecay=1,widthhidden=None, Lname='multiDense'):
    """Construct a basic NN with some dense layers.
    
    :parameter Nin: The number of inputs
    :type Nin: int
    :parameter Nout: The number of outputs
    :type Nout: int
    :parameter Nhidden: The number of hidden layers.
    :type Nhidden: int
    :parameter NDrop: The drop rate of the hidden layers.
    :type NDrop: int
    :parameter Ndecay: How quickly the layers should shrink.
    :type Ndecay: int
    :parameter widthhidden: The width of each hidden layer.
        If left at None, Nin + Nout will be used.
    :returns: The NN model
    :rtype: keras.Model
    
    """
    if widthhidden is None :
        if type(Nin) is list:
            neurons = Nin[1] + Nout
        else:
            neurons = Nin + Nout
    else:
        neurons = widthhidden
    x = inputs = keras.Input(shape=Nin, name=Lname+'_input')
    for i in range(Nhidden):
        if i == 0:
            x = layers.Dense(neurons, 
                             bias_initializer='ones', 
                             kernel_initializer = 'he_uniform',
                             name=Lname+'_dense0')(x)
            x = tf.nn.swish(x)
#             if Lname == 'Gbase' or Lname == 'Gft':
#                 x = layers.GaussianNoise(0.2)(x)
            x = layers.BatchNormalization(momentum=0.9, axis=1)(x)
        else:
            if NDrop < 1:
                x = layers.Dropout(NDrop)(x)
            neurons = np.round(neurons/Ndecay,0)
            x = layers.Dense(neurons,
                             kernel_initializer = 'he_uniform',
                             name=Lname+'_dense'+str(i))(x)
            x = tf.nn.swish(x)
#             if Lname == 'Gbase' or Lname == 'Gft':
#                 x = layers.GaussianNoise(0.2)(x)
            x = layers.BatchNormalization(momentum=0.9, axis=1)(x)
#     outputs = layers.Dense(Nout, activation='linear',name='multiDense_output')(x)
    outputs = layers.Dense(Nout,name=Lname+'_output')(x)
    outputs = tf.nn.swish(outputs)
    # tf.nn.swish(x)
    # tf.nn.leaky_relu(outputs, alpha=0.05)
    return keras.Model(inputs=inputs, outputs=outputs, name=Lname)

# used to do the weighted sum over views
def parallelwrapper(Nparallel,basemodel,insteadmax=False):
    """Construct a model that applies a basemodel multiple times and take a weighted sum 
    (or max) of the result.
    
    :parameter Nparallel: The number of times to apply in parallel
    :type Nparallel: int
    :parameter basemodel: a keras.Model inferred to have Nin inputs and Nout outputs.
    :type basemodel: a keras.Model
    :parameter insteadmax: If True, take the max of the results of the basemodel instead of the weighted sum.
        For compatibility, the model is still constructed with weights as inputs, but it ignores them.
    :type insteadmax: Boolean
    :returns: model with inputs shape [(?,Nparallel),(?,Nin,Nparallel)] and outputs shape (?,Nout).
        The first input is the scalar weights in the sum.
    :rtype: keras.Model
    
    Note: We could do a max over the parallel applications instead of or in addition to the weighted sum.
    
    """
    # infer shape of basemodel inputs and outputs
    Nin =  basemodel.inputs[0].shape[2]
    Nout =  basemodel.outputs[0].shape[1]
    
    # Apply basemodel Nparallel times in parallel
    # create main input (?,Nparallel,Nin) 
    parallel_inputs = keras.Input(shape=(Nparallel,Nin), name='parallelwrapper_input0')
    # apply base NN to each parallel slice; outputs (?,Nparallel,Nout)
    xb = basemodel(parallel_inputs)
    
    # create input scalars for weighted sun (?,Nparallel)
    weight_inputs = keras.Input(shape=(Nparallel,), name='parallelScalars')
    if insteadmax:
        # take max over the Nparallel direction to get (?,1,Nout)
        out = layers.MaxPool1D(pool_size=Nparallel)(xb)
        # reshape to (?,Nout)
        out = layers.Reshape((Nout,))(out)
    else:
        # do a weighted sum over the Nparallel direction to get (?,Nout)
        out = layers.Dot((-2,-1),name='weighted_sum_over_views')([xb,weight_inputs])
    
    return keras.Model(inputs=[weight_inputs,parallel_inputs], outputs=out, name='parallelwrapper')

# make models
# return generator NN
def init_generator(chem_data,tox_data,Nparameters):
    """Initialize the discriminator and generator cGAN neural nets.
    
    :returns: return generator and descriminator NN.
    :rtype: keras.Model
    
    """
    ## Option changing how results of each view are aggregated
    insteadmax = False # Does weighted average; original design
    #insteadmax = True # Does max instead of weighted average (for both G and D)
    Gfeatures = Nparameters[0] # number of chemical features produced by the generator
    Gbaselayers = Nparameters[1] # number of hidden layers in the generator used to extract features from views
    Glayers = Nparameters[2] # number of hidden layers in the generator used gen toxicity from features

    # G
    # base NN
    inShape = [chem_data[1].shape[1], chem_data[1].shape[2]]
    Gbase = multiDense(inShape,Gfeatures,Gbaselayers, Ndecay=1, NDrop=0, Lname='Gbase') # last args are hidden layers & drop rate
    # parallel view wrapper
    Gpw = parallelwrapper(chem_data[1].shape[1],Gbase,insteadmax)
#     print("Gpw output:", Gpw.outputs)
       
    # features to toxicity
    Gft = multiDense(Gfeatures,tox_data.shape[1],Glayers,Ndecay=1, NDrop=0, Lname='Gft') # last args are hidden layers & drop rate
#     print("Gft output:", Gft.outputs)
    # string together
    generator = keras.Model(inputs=Gpw.inputs,outputs=Gft(Gpw.outputs),name='generator') 
    # make trainable
    generator.compile(optimizer=generator_optimizer,loss=Gloss_function)
#     generator.summary()


    if 0:
        # sanity checks that model is working
        print("Sanity check:")
        gbv0call = Gbase(vs[:,0,:]).numpy()
        gbv0predict = Gbase.predict(vs[:,0,:])
        print("0 ?==", np.linalg.norm(gbv0call-gbv0predict))
        gpwcall = Gpw([ws,vs]).numpy()
        gpwpredict = Gpw.predict([ws,vs])
        print("0 ?==",np.linalg.norm(gpwcall-gpwpredict))
        gencall = generator([ws,vs]).numpy()
        genpredict = generator.predict([ws,vs])
        print("0 ?==",np.linalg.norm(gencall-genpredict))
        
    return generator

# make models
# return generator and discriminator NN
def init_NN(chem_data,tox_data, Nparameters):
    """Initialize the discriminator and generator cGAN neural nets.
    
    :parameter chem_data: The number of times to apply in parallel
    :type Nparallel: int
    :parameter tox_data: a keras.Model inferred to have Nin inputs and Nout outputs.
    :type basemodel: a keras.Model
    :parameter Nparameters: 
    :type insteadmax: np.array
    
    :returns: return generator and descriminator NN.
    :rtype: keras.Model
    
    """
    ## Option changing how results of each view are aggregated
    insteadmax = False # Does weighted average; original design
    #insteadmax = True # Does max instead of weighted average (for both G and D)
    Gfeatures = Nparameters[0] # number of chemical features produced by the generator
    Gbaselayers = Nparameters[1] # number of hidden layers in the generator used to extract features from views
    Glayers = Nparameters[2] # number of hidden layers in the generator used gen toxicity from features
    Dfeatures = Nparameters[3] # number of chemical features produced by the descriminator
    Dbaselayers = Nparameters[4] # number of hidden layers in the generator used to extract features from views
    Dlayers = Nparameters[5] # number of hidden layers in the discriminator
    n_classes = Nparameters[9] # number of chemical classes
    if n_classes is None:
        useLabel = False # build cGAN with (True) or without (False) labels
    else:
        useLabel = True # build cGAN with (True) or without (False) labels

#     This needs to be implemented but I'm not sure where at this time
#     # Instantiate the base model (or "template" model).
#     # We recommend doing this with under a CPU device scope,
#     # so that the model's weights are hosted on CPU memory.
#     # Otherwise they may end up hosted on a GPU, which would
#     # complicate weight sharing.
#     with tf.device('/cpu:0'):
        
    # G
    # base NN
    inShape = [chem_data[1].shape[1], chem_data[1].shape[2]]
    Gbase = multiDense(inShape,Gfeatures,Gbaselayers, Ndecay=1, NDrop=0, Lname='Gbase')
    # parallel view wrapper
    Gpw = parallelwrapper(chem_data[1].shape[1],Gbase,insteadmax)
#     print("Gpw output:", Gpw.outputs)

    if useLabel: # include class labels as a 2nd channel of the feature map (from machinelearningmastery.com)
        # label input
        in_label = layers.Input(shape=(1,))
        # embedding for categorical input
        Gli = layers.Embedding(n_classes, 50)(in_label)
        # scale up to gpw output dimensions with linear activation
        Gli = layers.Dense(Gfeatures)(Gli)
        # reshape to additional channel
        Gli = layers.Reshape((Gfeatures, 1))(Gli)
        # matrix input
        Gft_in = keras.Input(shape=(Gfeatures,1))
        # concat label as a channel
        Gli_merge = layers.Concatenate()([Gft_in, Gli])
            
        # features to toxicity
        Gft = multiDense(Gli_merge.shape[1],tox_data.shape[1],Glayers,Ndecay=1, NDrop=0, Lname='Gft')
    #     print("Gft output:", Gft.outputs)
        # string together
        generator = keras.Model(inputs=Gpw.inputs+Gli,outputs=[Gft(Gli_merge), Gpw.outputs[0]],name='generator') 
        # make trainable
        generator.compile(optimizer=generator_optimizer,loss=Gloss_function)
    
    else:
       
        # features to toxicity
        Gft = multiDense(Gfeatures,tox_data.shape[1],Glayers,Ndecay=1, NDrop=0, Lname='Gft')
    #     print("Gft output:", Gft.outputs)
        # string together
        generator = keras.Model(inputs=Gpw.inputs,outputs=[Gft(Gpw.outputs), Gpw.outputs[0]],name='generator') 
        # make trainable
        generator.compile(optimizer=generator_optimizer,loss=Gloss_function)


    # D
    # toxicity inputs
    if useLabel: # include class labels as a 2nd channel of the feature map (from machinelearningmastery.com)
        # label input
        in_label = layers.Input(shape=(1,))
        # embedding for categorical input
        Dli = layers.Embedding(n_classes, 50)(in_label)
        # scale up to gpw output dimensions with linear activation
        Dli = layers.Dense(Dfeatures)(Dli)
        # reshape to additional channel
        Dli = layers.Reshape((Dfeatures, 1))(Dli)
        # matrix input
        Dft_in = keras.Input(shape=(Dfeatures,1))
        # concat label as a channel
        Dli_merge = layers.Concatenate()([Dft_in, Dli])
        
        inShape = [chem_data[1].shape[1], chem_data[1].shape[2]]
        Dbase = multiDense(inShape,Dfeatures,Dbaselayers, Ndecay=1, NDrop=0, Lname='Dbase')
        # parallel view wrapper
        Dpw = parallelwrapper(chem_data[1].shape[1],Dbase,insteadmax)
    #     print("tox_data shape:", tox_data.shape)
    
        toxicity_inputs = keras.Input(shape=(tox_data.shape[1],), name='toxicity_inputs')

        # concatenate with toxicity (?,Nfeatures+Ntoxicity)
        concatft = layers.Concatenate()([Dli_merge,toxicity_inputs])
#         concatft = layers.Concatenate()([Dpw.outputs[0],toxicity_inputs])
        # features and toxicity to judgement
        Dftj = multiDense(concatft.shape[1],1,Dlayers, Ndecay=1, NDrop=0, Lname='Dftj') # last arg is hidden layers
        # string together
        discriminator = keras.Model(inputs=Dpw.inputs+[toxicity_inputs]+[Dli],
                                    outputs=Dftj(concatft),name='discriminator')
        
    elif 1: # let discriminator determine salient chemical features
        inShape = [chem_data[1].shape[1], chem_data[1].shape[2]]
        Dbase = multiDense(inShape,Dfeatures,Dbaselayers, Ndecay=1, NDrop=0, Lname='Dbase')
        # parallel view wrapper
        Dpw = parallelwrapper(chem_data[1].shape[1],Dbase,insteadmax)
    #     print("tox_data shape:", tox_data.shape)
    
        toxicity_inputs = keras.Input(shape=(tox_data.shape[1],), name='toxicity_inputs')

        # concatenate with toxicity (?,Nfeatures+Ntoxicity)
        concatft = layers.Concatenate()([Dpw.outputs[0],toxicity_inputs])
#         concatft = layers.Concatenate()([Dpw.outputs[0],toxicity_inputs])
        # features and toxicity to judgement
        Dftj = multiDense(concatft.shape[1],1,Dlayers, Ndecay=1, NDrop=0, Lname='Dftj') # last arg is hidden layers
        # string together
        discriminator = keras.Model(inputs=Dpw.inputs+[toxicity_inputs],
                                    outputs=Dftj(concatft),name='discriminator')
    else: # use chemcial features from Gpw
        toxicity_inputs = keras.Input(shape=(tox_data.shape[1],), name='toxicity_inputs')
        # use Gpw feature
        Gpw_features = keras.Input(shape=(Gfeatures,), name='Gpw_features')
        # concatenate with toxicity with label
        concatft = layers.Concatenate()([Gpw_features, toxicity_inputs])
    #     print(concatft.shape)
        # features and toxicity to judgement
        Dftj = multiDense(concatft.shape[1],Dfeatures,Dlayers, Ndecay=1, NDrop=0, Lname='Dftj')
        # string together
        discriminator = keras.Model(inputs=[Gpw_features]+[toxicity_inputs],
                                    outputs=Dftj(concatft),name='discriminator')
 
    # make model runable on multiple GPUs
#     discriminator = multi_gpu_model(discriminator, gpus=2)
    # make trainable
    discriminator.compile(optimizer=discriminator_optimizer,loss=Dloss_function)

    if 0:
        # sanity checks that model is working
        print("Sanity check:")
        gbv0call = Gbase(vs[:,0,:]).numpy()
        gbv0predict = Gbase.predict(vs[:,0,:])
        print("0 ?==", np.linalg.norm(gbv0call-gbv0predict))
        gpwcall = Gpw([ws,vs]).numpy()
        gpwpredict = Gpw.predict([ws,vs])
        print("0 ?==",np.linalg.norm(gpwcall-gpwpredict))
        gencall = generator([ws,vs]).numpy()
        genpredict = generator.predict([ws,vs])
        print("0 ?==",np.linalg.norm(gencall-genpredict))
        
    return generator, discriminator

# make models
def chem_features(chem_data, Nparameters, Lname='feature'):
    """Initialize the discriminator and generator cGAN neural nets.
    
    :parameter chem_data: The number of times to apply in parallel
    :type Nparallel: int
    :parameter tox_data: a keras.Model inferred to have Nin inputs and Nout outputs.
    :type basemodel: a keras.Model
    :parameter Nparameters: 
    :type insteadmax: np.array
    
    :returns: return generator and descriminator NN.
    :rtype: keras.Model
    
    """
    ## Option changing how results of each view are aggregated
    insteadmax = False # Does weighted average; original design
    #insteadmax = True # Does max instead of weighted average (for both G and D)
    features = Nparameters[0] # number of chemical features produced by the generator
    baselayers = Nparameters[1] # number of hidden layers in the generator used to extract features from views
    layers = Nparameters[2] # number of hidden layers in the generator used gen toxicity from features
        
    # base NN
    inShape = [chem_data[1].shape[1], chem_data[1].shape[2]]
    base = multiDense(inShape,features,baselayers, Ndecay=1, NDrop=0, Lname=Lname+'_base')
    # parallel view wrapper
    pw = parallelwrapper(chem_data[1].shape[1],base,insteadmax)
#     print("Gpw output:", Gpw.outputs)
    feature_model = keras.Model(inputs=pw.inputs,outputs=pw.outputs[0], name=Lname+'_model') 
    # make trainable
    feature_model.compile(optimizer=generator_optimizer,loss=Gloss_function)
    
    return feature_model

# return generator and discriminator NN
def init_NN_v2(chem_data,tox_data, Nparameters):
    """Initialize the discriminator and generator cGAN neural nets.
    
    :parameter chem_data: The number of times to apply in parallel
    :type Nparallel: int
    :parameter tox_data: a keras.Model inferred to have Nin inputs and Nout outputs.
    :type basemodel: a keras.Model
    :parameter Nparameters: 
    :type insteadmax: np.array
    
    :returns: return generator and descriminator NN.
    :rtype: keras.Model
    
    """
    ## Option changing how results of each view are aggregated
    insteadmax = False # Does weighted average; original design
    #insteadmax = True # Does max instead of weighted average (for both G and D)
    Gfeatures = Nparameters[0] # number of chemical features produced by the generator
    Gbaselayers = Nparameters[1] # number of hidden layers in the generator used to extract features from views
    Glayers = Nparameters[2] # number of hidden layers in the generator used gen toxicity from features
    Dfeatures = Nparameters[3] # number of chemical features produced by the descriminator
    Dbaselayers = Nparameters[4] # number of hidden layers in the generator used to extract features from views
    Dlayers = Nparameters[5] # number of hidden layers in the discriminator
    n_classes = Nparameters[9] # number of chemical classes
    if n_classes is None:
        useLabel = False # build cGAN with (True) or without (False) labels
    else:
        useLabel = True # build cGAN with (True) or without (False) labels

#     This needs to be implemented but I'm not sure where at this time
#     # Instantiate the base model (or "template" model).
#     # We recommend doing this with under a CPU device scope,
#     # so that the model's weights are hosted on CPU memory.
#     # Otherwise they may end up hosted on a GPU, which would
#     # complicate weight sharing.
#     with tf.device('/cpu:0'):
        
    # G
    # base NN
    Gpw_model = chem_features(chem_data,[Gfeatures,Gbaselayers,Glayers], Lname='gen_chem_feature')

    if useLabel: # include class labels as a 2nd channel of the feature map (from machinelearningmastery.com)
        # label input
        Gin_label = keras.Input(shape=(1,), name='gen_class_label')
        # embedding for categorical input
        Gli = layers.Embedding(n_classes, 50, name='gen_class_embedding')(Gin_label)
        # scale up to gpw output dimensions with linear activation
        Gli = layers.Dense(Gfeatures, name='gen_class_dense')(Gli)
        # reshape to additional channel
        Gli = layers.Reshape((Gfeatures,))(Gli)
        # concat label as a channel
        Gli_merge = layers.Concatenate(name='gen_label_merge')([Gpw_model.outputs[0], Gli])
            
        # features to toxicity
        Gft = multiDense(Gli_merge.shape[1],tox_data.shape[1],Glayers,Ndecay=1, NDrop=0, Lname='Gft')
        # string together
        generator = keras.Model(inputs=Gpw_model.inputs+[Gin_label],outputs=[Gft(Gli_merge), Gpw_model.outputs[0]],name='generator') 
        # make trainable
        generator.compile(optimizer=generator_optimizer,loss=Gloss_function)
    
    else:
       
        # features to toxicity
        Gft = multiDense(Gfeatures,tox_data.shape[1],Glayers,Ndecay=1, NDrop=0, Lname='Gft')
    #     print("Gft output:", Gft.outputs)
        # string together
        generator = keras.Model(inputs=Gpw_model.inputs,outputs=[Gft(Gpw_model.outputs), Gpw_model.outputs[0]],name='generator') 
        # make trainable
        generator.compile(optimizer=generator_optimizer,loss=Gloss_function)


    # D
    # toxicity inputs
    if 1: # use chemcial features from Gpw
        toxicity_inputs = keras.Input(shape=(tox_data.shape[1],), name='toxicity_inputs')
        # use Gpw feature
        Dpw_features = keras.Input(shape=(Gfeatures,), name='Gpw_features')

        if useLabel: # include class labels in the feature map (from machinelearningmastery.com)
            # label input
            Din_label = keras.Input(shape=(1,), name='disc_class_label')
            # embedding for categorical input
            Dli = layers.Embedding(n_classes, 50, name='disc_class_emmbedding')(Din_label)
            # scale up to gpw output dimensions with linear activation
            Dli = layers.Dense(Dfeatures, name='disc_class_dense')(Dli)
            # reshape to additional channel
            Dli = layers.Reshape((Dfeatures,))(Dli)
            # concat label as a channel
#             Dli_merge = layers.Concatenate()([Dpw.outputs[0], Dli])

            # concatenate with toxicity (?,Nfeatures+Ntoxicity)
            concatft = layers.Concatenate(name='disc_label-tox_merge')([Dpw_features, Dli ,toxicity_inputs])
    #         concatft = layers.Concatenate()([Dpw.outputs[0],toxicity_inputs])
            # features and toxicity to judgement
            Dftj = multiDense(concatft.shape[1],1,Dlayers, Ndecay=1, NDrop=0, Lname='Dftj') # last arg is hidden layers
            # string together
            discriminator = keras.Model(inputs=[Dpw_features]+[toxicity_inputs]+[Din_label],
                                        outputs=Dftj(concatft),name='discriminator')
        
    else: # let discriminator determine salient chemical features
        Dpw_model = chem_features(chem_data,[Dfeatures,Dbaselayers,Dlayers], Lname='disc_chem_feature')

        toxicity_inputs = keras.Input(shape=(tox_data.shape[1],), name='toxicity_inputs')
        
        # concatenate with toxicity (?,Nfeatures+Ntoxicity)
        concatft = layers.Concatenate()([Dpw_model.outputs[0],toxicity_inputs])
        # features and toxicity to judgement
        Dftj = multiDense(concatft.shape[1],1,Dlayers, Ndecay=1, NDrop=0, Lname='Dftj') # last arg is hidden layers
        # string together
        discriminator = keras.Model(inputs=Dpw_model.inputs+[toxicity_inputs],
                                    outputs=Dftj(concatft),name='discriminator')
 
    # make model runable on multiple GPUs
#     discriminator = multi_gpu_model(discriminator, gpus=2)
    # make trainable
    discriminator.compile(optimizer=discriminator_optimizer,loss=Dloss_function)

    if 0:
        # sanity checks that model is working
        print("Sanity check:")
        gbv0call = Gbase(vs[:,0,:]).numpy()
        gbv0predict = Gbase.predict(vs[:,0,:])
        print("0 ?==", np.linalg.norm(gbv0call-gbv0predict))
        gpwcall = Gpw_model([ws,vs]).numpy()
        gpwpredict = Gpw_model.predict([ws,vs])
        print("0 ?==",np.linalg.norm(gpwcall-gpwpredict))
        gencall = generator([ws,vs]).numpy()
        genpredict = generator.predict([ws,vs])
        print("0 ?==",np.linalg.norm(gencall-genpredict))
        
    return generator, discriminator

## based on tutorial, modified to cGAN
## modified to allow training of G or D alone

def discriminator_loss(real_output, fake_output):
    """This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.

        :parameter: real_output. predicted class using real toxicity matrix and chemical data
            :type: np.array
        :parameter: fake_output. predicted class using generated toxicity matrix and chemical data
            :type: np.array
    
        :returns: return total descriminator loss.
        :rtype: tuple
    """
    real_loss = Dloss_function(tf.ones_like(real_output), real_output)
    fake_loss = Dloss_function(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.

        :parameter: fake_output. vector with the weights.
            :type: np.array
    
        :returns: return generator loss.
        :rtype: tuple
    """
    return Gloss_function(tf.ones_like(fake_output), fake_output)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true+1), np.array(y_pred+1)
    return np.mean(np.abs(((y_true - y_pred)+1) / (y_true+1))) * 100
def mean_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true+1), np.array(y_pred+1)
    return np.mean(((y_true - y_pred)+1) / (y_true+1)) * 100
    
# Create wrapper function to allow model to be discarded and re-initilazation between cross-validation folds
def get_train_function():
    # Compile training function
    @tf.function
    def train_step(G_data,real_data,chemClass,toxClass,repeat,doG=True,doD=True):
        """Train Condictional Generative Adversarial Network.

        :parameter G_data: List containing a np.array vector with weights and 
                           a np.array matrix with vectorized views.
                           (see chemdataprep.load_pdb())
        :type G_data: list
        :parameter real_data: Master toxicity data matrix. 
                              Rows correspond to chemicals and columns to toxicity measurements.
                              (see toxmathandler.load_tmats())
        :type real_data: np.array
        :parameter chemClass: Chemical class label.
        :type chemClass: np.array (int)
        :parameter toxClass: Toxcicity class labels.
        :type toxClass: np.array (int)
        :parameter repeat: Number of repeats per label
        :type repeat: np.array (int)
        :parameter doG: If True train generator
        :type doG: boolean
        :parameter doD: If True train discriminator
        :type doD: boolean

        :returns: Discriminator and Generator loss
        :rtype: tuple

        """
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_matrix, Gpw_features = generator(G_data+chemClass, training=True)
            expanded_Dpw_model = tf.repeat(Gpw_features, repeats=repeat, axis=0)

            real_output = discriminator([expanded_Dpw_model]+real_data+[toxClass], training=True)
            fake_output = discriminator([Gpw_features]+[generated_matrix]+chemClass, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            if doD:
                # update discriminator
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

                # Additional training
                for i in range(2):
                    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                        generated_matrix, Gpw_features = generator(G_data+chemClass, training=False)
                        expanded_Dpw_model = tf.repeat(Gpw_features, repeats=repeat, axis=0)

                        real_output = discriminator([expanded_Dpw_model]+real_data+[toxClass], training=True)
                        fake_output = discriminator([Gpw_features]+[generated_matrix]+chemClass, 
                                                    training=True)


                        gen_loss = generator_loss(fake_output)
                        disc_loss = discriminator_loss(real_output, fake_output)

                    # update discriminator
                    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            if doG:
                # update generator
                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        return (gen_loss,disc_loss)
    return train_step


# check judgements of D & G
# G wants y-axis to be 1
# D wants y-axis to be 0 and the x-axis to be 1
def judgement_plot(Nws, Nvs, Ntoxicity, Ngenerator, Ndiscriminator, Nlegend, Nparameters=None,
                   showLoss=0, saveImages=None, dType="Training", verbose=0):
    """Plot the generator (y-axis) and discriminator (x-axis) loss functions. 
        Construct a plot of the true data (x-axis) vs the data generated by the cGAN (y-axis).

        :parameter: Nws. vector with the weights.
            :type: np.array
        :parameter: Nvs. matrix with the vectorized views. Each row is a view.
            :type: np.array
        :parameter: Ntoxicity. vector representing the real toxicity data.
            :type: np.array
        :parameter: Ngenerator. tf.keras generator model.
            :type: tf.keras.model
        :parameter: Ndiscriminator. tf.keras discriminator model.
            :type: tf.keras.model
        :parameter: Nlegend. The string labels corresponding to the columns of labels 
            :type legend: None or list of str
        :parameter: Nparameters. Neural netowrk and views parameters
            :type np.array
    
    """
    gen_tox, chem_features = Ngenerator.predict([Nws, Nvs])
    gen_ability = tf.sigmoid(Ndiscriminator.predict([Nws, Nvs,gen_tox])).numpy()
    disc_ability = tf.sigmoid(Ndiscriminator.predict([Nws, Nvs,Ntoxicity])).numpy()
    model_MAPE_T = mean_absolute_percentage_error(Ntoxicity, gen_tox)
    MSE = np.square(np.subtract(Ntoxicity,gen_tox)).mean() 
    figRows=2
    figHeight=8

    if showLoss:
        figRows=3
        figHeight=13
        
    fig = plt.figure(1, figsize=[13,figHeight])
    # set up subplot grid
    gridspec.GridSpec(figRows,2)

    ax = plt.subplot2grid((figRows,2), (0,0), colspan=1, rowspan=1)
#     ax = plt.subplot(121)
    if verbose:
        if Nparameters is not None:
            ax.set_title("Chemical Information:\n"+
                         "Data tensor (w,v) shapes = "+str(Nws.shape)+" "+str(Nvs.shape)+
                         "\nOnly use Carbon atoms: "+str(Nparameters[6])+
                         "\nNumber of atoms per view: "+str(Nparameters[7])+
                         "\nViews per chemical: "+str(Nparameters[8])+
                         "\n\nNetwork Information:"+
                         "\nGfeatures = "+str(Nparameters[0])+"\nGbaselayers = "+str(Nparameters[1])+
                         "\nGlayers = "+str(Nparameters[2])+"\nDfeatures = "+str(Nparameters[3])+
                         "\nDbaselayers = "+str(Nparameters[4])+"\nDlayers = "+str(Nparameters[5])+
                         "\n\n"+dType+" Dataset"+
                         "\nMean Absolute Percent Error: "+str(model_MAPE_T)+
                         "\nMean Squared Error: "+str(MSE), loc='left')
        else:
            ax.set_title("\n\n"+dType+" Dataset"+
                         "\nMean Absolute Percent Error: "+str(model_MAPE_T)+
                         "\nMean Squared Error: "+str(MSE), loc='left')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.plot(disc_ability,gen_ability,'o')
    plt.xlabel('Discriminator Ability')
    plt.ylabel('Generator Ability')
    
    plt.subplot2grid((figRows,2), (0,1), colspan=1, rowspan=1)
    plt.xlabel('Real Values')
    plt.ylabel('Generated Vaues')

    if 1: 
        # optional reference line
        mintrue = np.min(Ntoxicity)
        maxtrue = np.max(Ntoxicity)
        plt.plot([mintrue,maxtrue],[mintrue,maxtrue])
    for i in range(Ntoxicity.shape[1]):
        # include labels
        plt.plot(Ntoxicity[:,i],gen_tox[:,i],'o', 
                 label = Nlegend[0][Nlegend[2][i//len(Nlegend[3])]]+
                 ' '+Nlegend[0][Nlegend[3][i%len(Nlegend[3])]])

    plt.subplot2grid((figRows,2), (1,0), colspan=2, rowspan=1)
    plt.xlabel('Real Values')
    plt.ylabel('Residuals')
    for i in range(Ntoxicity.shape[1]):
        # Plot the residuals
        obs = gen_tox[:,i]
        exp = Ntoxicity[:,i]
        residuals = np.subtract(obs, exp)
        # include labels
        plt.plot(Ntoxicity[:,i],residuals,'o', 
                 label = Nlegend[0][Nlegend[2][i//len(Nlegend[3])]]+
                 ' '+Nlegend[0][Nlegend[3][i%len(Nlegend[3])]])
        
    if showLoss:    
        # plot loss during training
        plt.subplot2grid((figRows,2), (2,0), colspan=1, rowspan=1)
        plt.title('Loss / Mean Absolute Percent Error')
        plt.plot(training_loss, label='train')
        plt.plot(val_loss, label='validate')
        plt.legend()
        plt.show()
    if saveImages is not None:
        plt.savefig(saveImages+'.png', dpi=600, bbox_inches='tight')
    plt.pause(0.5)

# plot predictions of G versus truth
def PvT_plot(model,data,labels,legend=None,title=None,doresidual=False, Nparameters=None,
             saveImages=None,verbose=0):
    """Construct a plot of the true labels (x-axis) vs the data generated by the model (y-axis).
    
    :parameter model: the model (e.g. NN)
    :type model: keras.model
    :parameter data: the data that can be input to the model
    :type data: numpy.array
    :parameter labels: the true outputs corresponding to the data
    :type labels: numpy.array
    :parameter legend: The string labels corresponding to the columns of labels 
    :type legend: None or list of str
    :parameter title: A title for the plot
    :type title: None or string
    :parameter doresidual: If true, plot the residual instead
    :type doresidual: Boolean
    :parameter saveImages: path+filename of output image without file extension (png is default)
    :type saveImages: str
    """
        
    gen_lab = model.predict(data)
    if doresidual:
        gen_lab = gen_lab - labels


    MSE = np.square(np.subtract(labels,gen_lab)).mean()
    MAPE = mean_absolute_percentage_error(labels,gen_lab)
    
    plt.figure()
    ax = plt.subplot(111)        
    if verbose:
        if Nparameters is not None:
            ax.set_title("Chemical Information:\n"+
                         "Data tensor (w,v) shapes = "+str(data[0].shape)+" "+str(data[1].shape)+
                         "\nOnly use Carbon atoms: "+str(Nparameters[3])+
                         "\nNumber of atoms per view: "+str(Nparameters[4])+
                         "\nViews per chemical: "+str(Nparameters[5])+
                         "\n\nNetwork Information:"+
                         "\nGfeatures = "+str(Nparameters[0])+"\nGbaselayers = "+str(Nparameters[1])+
                         "\nGlayers = "+str(Nparameters[2])+
                         "\n\n"+title+" Dataset"+
                         "\nMean Absolute Percent Error: "+str(MAPE)+
                         "\nMean Squared Error: "+str(MSE), loc='left')
        else:
            ax.set_title("\n\n"+title+" Dataset"+
                         "\nMean Absolute Percent Error: "+str(MAPE)+
                         "\nMean Squared Error: "+str(MSE), loc='left')
    if legend is None:
        for i in range(labels.shape[1]):
            plt.plot(labels[:,i],gen_lab[:,i],'o')
    else:
        print(labels.shape[1])
        for i in range(labels.shape[1]):
            # include legend
            plt.plot(labels[:,i],gen_lab[:,i],'o', label=legend[i])
        ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    ax.set_xlabel('True Values')
    if doresidual:
        ax.set_ylabel('Residual Values')
    else:
        ax.set_ylabel('Generated Values')
        # reference line
        mintrue = np.min(labels)
        maxtrue = np.max(labels)
        plt.plot([mintrue,maxtrue],[mintrue,maxtrue])
#     if title is not None:
#         plt.title(title)
    if saveImages is not None:
        plt.savefig(saveImages+'.png', dpi=600, bbox_inches='tight')
    plt.pause(0.5)
    
# plot predictions of G versus truth
def Gen_PvT_plot(data, pred, true,legend=None,title=None,doresidual=False, Nparameters=None,
             saveImages=None,verbose=0):
    """Construct a plot of the true true (x-axis) vs the data generated by the model (y-axis).
    
    :parameter model: the model (e.g. NN)
    :type model: keras.model
    :parameter data: the data that can be input to the model
    :type data: numpy.array
    :parameter true: the true outputs corresponding to the data
    :type true: numpy.array
    :parameter legend: The string labels corresponding to the columns of labels 
    :type legend: None or list of str
    :parameter title: A title for the plot
    :type title: None or string
    :parameter doresidual: If true, plot the residual instead
    :type doresidual: Boolean
    """
        
    if doresidual:
        pred = pred - true

    MSE = np.square(np.subtract(true,pred)).mean()
    MAPE = mean_absolute_percentage_error(true,pred)
    
    plt.figure()
    ax = plt.subplot(111)        
    if verbose:
        if Nparameters is not None:
            ax.set_title("Chemical Information:\n"+
                         "Data tensor (w,v) shapes = "+str(data[0].shape)+" "+str(data[1].shape)+
                         "\nOnly use Carbon atoms: "+str(Nparameters[6])+
                         "\nNumber of atoms per view: "+str(Nparameters[7])+
                         "\nViews per chemical: "+str(Nparameters[8])+
                         "\n\nNetwork Information:"+
                         "\nGfeatures = "+str(Nparameters[0])+"\nGbaselayers = "+str(Nparameters[1])+
                         "\nGlayers = "+str(Nparameters[2])+"\nDfeatures = "+str(Nparameters[3])+
                         "\nDbaselayers = "+str(Nparameters[4])+"\nDlayers = "+str(Nparameters[5])+
                         "\n\n"+title+" Dataset"+
                         "\nMean Absolute Percent Error: "+str(MAPE)+
                         "\nMean Squared Error: "+str(MSE), loc='left')
        else:
            ax.set_title("\n\n"+title+" Dataset"+
                         "\nMean Absolute Percent Error: "+str(MAPE)+
                         "\nMean Squared Error: "+str(MSE), loc='left')
    if legend is None:
        for i in range(len(true)):
            plt.plot(true[i],pred[i],'o')
    else:
#         print(true.shape[1])
        for i in range(len(true)):
            # include legend
            plt.plot(true[i],pred[i],'o')
#         ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    ax.set_xlabel('True Values')
    if doresidual:
        ax.set_ylabel('Residual Values')
    else:
        ax.set_ylabel('Generated Values')
        # reference line
        mintrue = np.min(true)
        maxtrue = np.max(true)
        plt.plot([mintrue,maxtrue],[mintrue,maxtrue])
#     if title is not None:
#         plt.title(title)
    if saveImages is not None:
        plt.savefig(saveImages+'.png', dpi=600, bbox_inches='tight')
    plt.pause(0.5)    
    
def write_training_file(Lparameters,Lmodel,Lmetrics,Lfilename):
    """Writing training parameters and training results to an excel file.
    
    :parameter Lparameters: cGAN & views parameters - 
                            [Gfeatures, Gbaselayers, Glayers, 
                            Dfeatures, Dbaselayers,Dlayers,
                            carbonbased, setNatoms, views, ClassLabels]
    :type Lparameters: list
    
    :parameter Lmodel: Model details - 
                         [Model_name, concentrations, views, gen_ability_V, disc_ability_V]
    :type Lmodel: np.array
    
    :parameter Lmetrics: Confusion matrix metrics and accuracy { display_conf_matrix() output } 
                         [NN_kappa_score, NN_auroc, SE, SP, PPV]
    :type Lmetrics: np.array
    
    :parameter Lfilename: The number of hidden layers.
    :type Lfilename: str

    :returns: Summary sheet
    :rtype: DataFrame
    
    """
    LGparm = [str(Lparameters[0]),str(Lparameters[1]),str(Lparameters[2])]
    LGparm = " ".join(LGparm)
    LDparm = [str(Lparameters[3]),str(Lparameters[4]),str(Lparameters[5])]
    LDparm = " ".join(LDparm)
    sheet = 'Train-Val-iGAN'
    
    my_file = pathlib.Path(Lfilename)
    if my_file.is_file():
        # file exists
        try:
            summary_file = pd.ExcelFile(my_file)
        except:
            summary_file = pd.DataFrame(columns=['ID','Number of Conc.', 'Views', 'Atoms per View',
                                                 'Gen Parameters', 'Disc Parameters', 
                                                 'Gen Ability - Val','Disc Ability - Val',
                                                 'SE', 'SP', 'PPV', 'Kappa', 'AUROC'])        
    else:
        # if no file
        summary_file = pd.DataFrame(columns=['ID','Number of Conc.', 'Views', 'Atoms per View',
                                                 'Gen Parameters', 'Disc Parameters', 
                                                 'Gen Ability - Val','Disc Ability - Val',
                                                 'SE', 'SP', 'PPV', 'Kappa', 'AUROC']) 
    
    parameter_output = [[Lmodel[0], len(Lmodel[1]),str(Lmodel[2]), str(Lparameters[7]), 
                         LGparm, LDparm, 
                         Lmodel[3], Lmodel[4],
                         Lmetrics[2], Lmetrics[3], Lmetrics[4], Lmetrics[0], Lmetrics[1]]]
    parameter_output_df = pd.DataFrame(parameter_output,columns=['ID','Number of Conc.', 'Views', 'Atoms per View',
                                                                 'Gen Parameters', 'Disc Parameters', 
                                                                 'Gen Ability - Val','Disc Ability - Val',
                                                                 'SE', 'SP', 'PPV', 'Kappa', 'AUROC'])

    if not isinstance(summary_file, pd.DataFrame):
        summary_file_df = pd.read_excel(summary_file, sheet_name=sheet)
    else:
        summary_file_df = summary_file
    summary_file_df = summary_file_df.append(parameter_output_df)

    with pd.ExcelWriter(my_file, mode='A') as writer:  # doctest: +SKIP
        summary_file_df.to_excel(writer,sheet_name=sheet, header=True, index=False)
        
    return summary_file_df
