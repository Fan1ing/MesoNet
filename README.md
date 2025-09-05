# MesoNet
####  MesoNet: A Fundamental Principle for Multi-Representation  Learning in Complex Chemical Systems
This work presents a highly interpretable machine learning model designed to elucidate complex intramolecular and intermolecular interactions, particularly in multicomponent mixtures. The model achieves unified predictive modeling for both binary and ternary mixture systems within a single framework, overcoming key challenges in handling compositional diversity and interaction nonlinearity.  

Key features include:  
1) **Physics-aware architecture** that preserves chemical interpretability while capturing interaction hierarchies  
2) **Unified representation learning** enabling seamless cross-component generalization  
3) **cross scale mechanisms** that quantify contribution weights of molecular subgroups  

The model demonstrates superior performance in predicting mixture properties while maintaining transparency - offering both accurate predictions and mechanistic insights into interaction. 


## Code running conditions

#MesoNet is implemented using Pytorch and runs on Ubuntu with NVIDIA GeForce RTX 4090 graphics processing units,which relies on Pytorch Geometric.

The following are the required Python libraries to be installed：numpy、pandas、rdkit、sklearn、ncps


## Data preparation
To facilitate the reproduction of our results, we have provided three example running codes for three components, two components, and one component respectively (this requires adjusting the CSV file path for the corresponding dataset to your local path). The code will generate the graph dataset (A file package containing a pt file) and then proceed to the next step of prediction. The code will output and print the errors for the training and test sets.

If you wish to use your own dataset for prediction, simply prepare the required molecular SMILES format and its properties for prediction, and then process the data into the same format as the one we provided.

## The following table shows an example of processing the activity coefficients of three components
|solv1_x|solv2_x|solv3_x|solv1_gamma|solv2_gamma|solv3_gamma|solv1_smiles|solv2_smiles|solv3_smiles|
|----------------|----------|----------|----------|----------|----------|----------|----------|----------|


## Running time

The generation of molecular graphs for single component and two-component datasets takes approximately a few minutes, while the activity coefficient, due to the large amount of data (greater than 100000), takes less than an hour to generate molecular graphs,all of which are created on a 10 core CPU.
For the running time of the model, it takes about several tens of hours to complete the complete five fold cross validation in the prediction of activity coefficients.
The prediction of other properties varies in time from tens of minutes to several hours.


## Hyperparameters and model prediction:
In our training, we did not perform hyperparameter tuning on the validation set for each individual attribute prediction and then make predictions on the test set. In single component, two-component, and three component prediction, we select the same hyperparameters (learning rate, number of neurons, network architecture, etc.) and directly use the training set for training and predicting the test set (validation set).

In model prediction, error printing is performed directly on the training and testing sets to observe the changes in training and prediction errors. Alternatively, the overall training error and best results can be saved.

ps： In practical applications, tuning hyperparameters for each property may increase the predictive performance of the model, but we did not do so because the training time was too long and the predictive performance was already better than the models reported in the literature.

## Continuously updated
We will continue to update data and models in the future，and we will constantly check the correctness and completeness of the code and data. If there are any code running errors or any questions, please contact my email: fanjinming@zju.edu.cn



