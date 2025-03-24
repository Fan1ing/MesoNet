# NGNN
####  A Natural Growth Model for Predicting Physicochemical Property of Complex Systems.
For more design concepts and details of the model, please refer to Article  ***A Universal Framework for General Prediction of Physicochemical Properties: the Natural Growth Model***

# Overview

Here are the details about the model.

## Code running conditions

#NGNN is implemented using Pytorch and runs on windows11 with NVIDIA GeForce RTX 3060 graphics processing units,which relies on Pytorch Geometric.

The following are the required Python libraries to be installed：numpy、pandas、rdkit、sklearn.

## Data preparation
If you need to use your own dataset for prediction, simply prepare the required molecular smiles format and its properties for prediction, and then process the data into the same format as the data we provide.

## Code content
|name |content |
|----------------|--------------------------------|
|**NGNN.m** | NGNN code used to predict multi solvent molecular properties. |
|**msMPNN.m** |NGNN code used to predict single solvent molecular properties.|

## Data content
|name |content |
|----------------|--------------------------------|
|**fanjinming.csv** | Absorption wavelength data. |
|**fanjinming1.csv** |Solubility data.|
|**data_E.csv** | Environmental features (solvent). |
|**data_G.csv** |Solvational interaction parameter.|

## Continuously updated
We will continue to update the data and models in the future.If you have any questions, please contact us. The contact information can be found in our paper.



