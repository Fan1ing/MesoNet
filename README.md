# MesoNet
####  A MesoNet Model for Predicting Physicochemical Property of Complex Systems.
For more design concepts and details of the model, please refer to Article  ***Non-Random Parameterized Networks for Cross-Scale Modeling of Compositional Interplay ***
![Model principle](picture/Fig1.png)

![Model principle](picture/Fig2.png)

# Overview

Here are the details about the model.


## Code running conditions

#MesoNet is implemented using Pytorch and runs on Ubuntu with NVIDIA GeForce RTX 4090 graphics processing units,which relies on Pytorch Geometric.

The following are the required Python libraries to be installed：numpy、pandas、rdkit、sklearn、ncps


## Data preparation
To facilitate the reproduction of our results, you only need to directly execute the code for each predicted property (this requires adjusting the CSV file path for the corresponding dataset to your local path). The code will generate the graph dataset and then proceed to the next step of prediction. The code will output and print the errors for the training and test sets.

If you wish to use your own dataset for prediction, simply prepare the required molecular SMILES format and its properties for prediction, and then process the data into the same format as the one we provided.

## Running time

The generation of molecular graphs for single component and two-component datasets takes approximately a few minutes, while the activity coefficient, due to the large amount of data (greater than 100000), takes less than an hour to generate molecular graphs,all of which are created on a 10 core CPU.

For the running time of the model, it takes about several tens of hours to complete the complete five fold cross validation in the prediction of activity coefficients.
The prediction of other properties varies in time from tens of minutes to several hours


## Continuously updated
We will continue to update data and models in the future. If you have any questions, please contact my email: fanjinming@zju.edu.cn



