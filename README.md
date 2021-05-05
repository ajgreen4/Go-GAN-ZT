# Leveraging high-throughput screening data, deep neural networks, and conditional generative adversarial networks to advance predictive toxicology

A combined deep neural network (DNN) and conditional Generative Adversarial Network (cGAN) can leverage a large chemical set of experimental toxicity data plus chemical structure information to predict the toxicity of untested compounds.

This repository contains four Jupyter Notebook files:
 - Go-ZT notebooks
 	- developRegGen-(0,1)_18_6_kfold.ipynb
 		- Code used to run a random 10 fold cross validation of the DNN
 	- developRegGen-(0,1)_18_6-final.ipynb
 		- Code used to train the final DNN model
 - GAN-ZT notebooks
 	- developcGAN-(0,1)_indv_18_1_kfold.ipynb
 		- Code used to run a random 10 fold cross validation of the cGAN
 	- developcGAN-(0,1)_indv_18_1_final.ipynb
 		- Code used to train the final cGAN model

 These files should be run in the Singularity container environment build specificaly for this project to ensure all dependencies are present and that the code runs error free. The container can be found at: https://cloud.sylabs.io/library/ajgreen/default/go_gan_zt_container
