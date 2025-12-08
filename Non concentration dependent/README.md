### Download datasets

* Download Chromophore dataset from https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2, and leave only **Absorption max (nm)**,  **Emission max (nm)**, and **Lifetime (ns)** column.

  * Make separate csv file for each column, and erase the NaN values for each column.
  * We log normalize the target value for **Lifetime** data .
  * Modify the corresponding labels of the dataset to the required format in data_processing and run it directly.

### code

&nbsp;	We have provided code examples for absorbing wavelengths.

