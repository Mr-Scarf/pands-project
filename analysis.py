# analysis.py

# Step 3 of Project
# 1. Outputs a summary of each variable to a single text file,  
# 2. Saves a histogram of each variable to png files, and  
# 3. Outputs a scatter plot of each pair of variables.  
# 4. Performs any other analysis you think is appropriate. 
# 
# 
# Author: David Scally

#Instructions - to be Deleted

                        #Minimum Viable Project 
                        #  
                        #   The minimum standard is a GitHub repository containing a README, a Python script, a 
                        #   generated summary text file, and images. The README (and/or notebook) should contain a 
                        #   summary of the data set and your investigations into it. It should also clearly document how 
                        #   to run the Python code and what that code does. Furthermore, it should list all references 
                        #   used in completing the project. 
 
                        #   A better project will be well organised and contain detailed explanations. The analysis will 
                        #   be well conceived, and examples of interesting analyses that others have pursued based on 
                        #   the data set will be discussed. Note that the point of this project is to use Python. You may 
                        #   use any Python libraries that you wish, whether they have been discussed in class or not. 
                        #   You should not be thinking of using spreadsheet software like Excel to do your calculations.  


# i -  Import libraries that will be required for the project

import sklearn as skl                                      # Machine learning Library that contains sample datasets
import sklearn.datasets                                    # datasets is submodule of sklearn module , used in this project
import matplotlib.pyplot as plt                            # Used for plotting graphs
import pandas as pd                                        # Used to analayse data
import numpy as np                                         # Used for large datasets to perform numerical computing
import sys                                                 # The 'sys' module provides functions and variables used to manipulate different parts of the Python runtime environment.



data = skl.datasets.load_iris()                           #Load the iris dataset from scikit-learn

df = pd.DataFrame(data.data, columns=data.feature_names)  # Convert to Pandas Dataframe

#print(data)

# Comments: sklearn used for import of iris dataset - covered in Principles of Data Analytics course.
# Reference https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset



# 1 - Output a summary of each variable to a single text file. 

# The four variables in the dataset are:
# - sepal length (cm) , sepal width (cm) , petal length (cm) , petal width (cm)

print(df.head())                                           # df.head shows us the first 5 rows of the dataset & the variable data that has been collected

print(df.shape)                                            # df.shape shows us the size of the dataset(rows & columns) that we are working with

print(df.describe())                                       # df.describe() is used to show statistical data about the Iris dataset such as Mean, Minimum, Maximum, Standard Deviation & Median


# Comments: df head/shape/desribe - covered in Principles of Data Analytics course.
# Reference 



# 2 - Saves a histogram of each variable to png files - 
# 6th April Comment - Add link from datacamp & use ref in D.A project. iloc not requires as I have not added target names as numerical column in this modeule as of yet(Review if required & to keep consistenet)

print(df.columns)

#From Data Analyics course - Applicable columns are first 4, dataset will include the added columns of species', 'species_name' so subset needed for 4 feautures
# iloc used to create subset for required features only format is row,columns
df_subset_features=df.iloc[:, 0:4]

# Plot histogram of Features.
df_subset_features.hist(bins =20,alpha = .9 , color='skyblue', edgecolor='black')


# Add axis lables
plt.xlabel('Value')
plt.ylabel('Count')

# Set the title for all histograms - Reference: SuperTitle:  https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html
plt.suptitle("Histogram : Features of Iris Dataset",fontweight='bold')  # Supertitle for all histograms

# Show the plot
plt.show()








# 3. Outputs a scatter plot of each pair of variables.  


