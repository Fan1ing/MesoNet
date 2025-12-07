\# MesoNet

\####  A MesoNet Model for Predicting Physicochemical Property of Complex Systems.

!\[MesoNet Architecture](./figures/MesoNet.tif)



\## Code running conditions



\#MesoNet is implemented using Pytorch and runs on Ubuntu with NVIDIA GeForce RTX 4090 graphics processing units,which relies on Pytorch Geometric.



The following are the required Python libraries to be installed：numpy、pandas、rdkit、sklearn、ncps



In the Concentration dependent directory, we provide:



Detailed implementation of each module



Data generation scripts



Example usage based on the activity coefficient dataset



These examples can help you understand the workflow of data preprocessing and model training.

\## Hyperparameters and model prediction:

In our training, we did not perform hyperparameter tuning on the validation set for each individual attribute prediction and then make predictions on the test set. 



ps： In practical applications, tuning hyperparameters for each property may increase the predictive performance of the model, but we did not do so because the training time was too long and the predictive performance was already better than the models reported in the literature.

