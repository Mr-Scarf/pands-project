

# Principles of Data Analytics - Iris Flower Analysis

**Author**: David Scally  
**University**: Atlantic Technological University  
**Module**: Programming and Scripting 
**Class**: January 2025 

This repository contains an analysis of the Iris flower dataset using Python and Jupyter Notebook, as part of the Principles of Programming and Scripting module.

## Iris Flowers
![Iris Flowers](https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png)


## Purpose

This is an analysis of the famous Iris dataset by the British biologist Ronald Fisher. The Iris dataset is a great entry-level dataset that allows us to explore relationships between features of iris flowers and practice data analysis techniques.

It contains 150 iris flower samples from 3 different species & 4 features per sample.

**Species of iris flower:**
 - Setosa
 - Versicolor
 - Virginica

**Features of iris flower:**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width


The module project requirements are set out below. 

1. Research the data set online and write a summary about it in your README.  
2. Download the data set and add it to your repository.  
3. Write a program called analysis.py that:  
  1. Outputs a summary of each variable to a single text file,  
  2. Saves a histogram of each variable to png files, and  
  3. Outputs a scatter plot of each pair of variables.  
  4. Performs any other analysis you think is appropriate. 


In addition, we will create a Jupyter notebook 'Analysis.ipynb' to group our code & analysis.

## Repository Contents

 - README.md - This file, used to explain the project
 - iris.dataset - Downloaded raw dataset from the UC Irvine (UCI) Machine Learning Repository Archive - https://archive.ics.uci.edu/dataset/53/iris
 - analysis.ipynb - Jupyter notebook for Data Analysis of Iris Dataset
 - Iris Dataset - Summary Statistics.txt - Output of iris data summary statistics saved in a text file
 - Histograms - Output of iris dataset variable histograms saved in a png file
 - analysis.py - Python file with code used in analysis
 - requirements.txt - List of packages used in the analysis
 - gitignore - Lists files that git should ignore



## Structure of Analysis

We will explore the dataset by following the below steps in the Jupyter notebook 'Analysis.ipynb' :

1.  **Summary of the Dataset & explain the Python libraries Used**
List and brief explanation of python libraries that will be used to explore the iris dataset.

2.  **Source & Explore the Iris dataSet**
The original iris dataset can be downloaded as a csv file format from the UCI website - https://archive.ics.uci.edu/dataset/53/iris
We have stored a csv copy in the repository for reference - see 'Iris datset' but we also can use the python libaries built in loader , **scikit-learn**, to import the data set directly into the Jupyer notebook - we will use the latter method.
The Iris dataset is loaded using `sklearn.datasets.load_iris()` and converted into a pandas DataFrame. A comparison between the raw dataset from sklearn vs converted pandas DataFrame
Initial inspection includes checking data types, size, and a preview of the dataset using `df` and `.info()`.


3.  **Summarise the Data**
Descriptive statistics (mean,min,max,standard deviation,median) are generated using `.describe()` & saved to a text file.

4.  **Visualise Features -Histogram**
Used `matplotlib` to create histograms to understand the distribution of each feature & saved to a png file.

5.  **Investigate *& Analyse Relationships - Scatter Plot**
Used 'seaborn' to create a pairplot of the features & discuss what the pairplot depicts.
Created an additional scatter plot of two features - 'petal length (cm)' & 'petal width (cm)' - and added a regression line using 'numpy.polyfit'  to show the distribution of the data.

6.  **Other Analysis - Fit a Simple Linear Regression**
Calculated the coefficient of determination r²

7.  **Analyse Class Distributions**
Create box-plots of the petal lengths for each of the three classes using color-coded plots.

8.  **Other Analysis - Compute Correlations**
Calculate the correlation coefficients between the features & displayed the results as a heatmap using 'matplotlib'.

9. **Conclusion**
Closing comments on the analysis & key findings.


  ## Installation

1. Clone the repository

git clone https://github.com/Mr-Scarf/principles_of_data_analytics.git

2. Install dependencies - see file 'requirements.txt'

pip install -r requirements.txt

3. Open file `tasks.ipynb` Jupyter notebook  and run the code cells.



## References

- [GeeksforGeeks: Iris Dataset Overview](https://www.geeksforgeeks.org/iris-dataset/)
- [Curran's Gist: Iris Dataset Visualization](https://gist.github.com/curran/a08a1080b88344b0c8a7)







# pands-project
Programming And Scripting - Project

## Problem Statement/Aim 


This project concerns the well-known Fisher’s Iris data set [3]. You must research the data set 
and write documentation and code (in Python [1]) to investigate it. An online search for 
information on the data set will convince you that many people have investigated it 
previously. You are expected to be able to break this project into several smaller tasks that 
are easier to solve, and to plug these together after they have been completed.  
You might do that for this project as follows:  
1. Research the data set online and write a summary about it in your README.  
2. Download the data set and add it to your repository.  
3. Write a program called analysis.py that:  
    (i) Outputs a summary of each variable to a single text file,  
    (ii) Saves a histogram of each variable to png files, and  
    (iii) Outputs a scatter plot of each pair of variables.  
    (iv)   Performs any other analysis you think is appropriate. 
You may produce a Jupyter notebook as well containing all your comment. This notebook 
should only contain text that you have written yourself, (it may contain referenced code 
from other sources). I will harshly mark any text (not code) that I feel is not written directly 
by you. I want to know what YOU think, not some third party. Please make sure the style of 
your documentation is consistent. 
It might help to suppose that your manager has asked you to investigate the data set, with a 
view to explaining it to your colleagues. Imagine that you are to give a presentation on the 
data set in a few weeks’ time, where you explain what investigating a data set entails and how 
Python can be used to do it. You have not been asked to create a deck of presentation slides, 
but rather to present your code and its output to them. 