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


## Task 1: Summary of the Dataset & explain the Python libraries Used

import sklearn as skl                                      # Scikit-learn (sklearn) - Machine learning Library that contains sample datasets - used in this project to load the iris dataset
import sklearn.datasets                                    # Scikit-learn datasets  - specific sub- module of sklearn library used to load the iris datatset
import matplotlib.pyplot as plt                            # Matplotlib - Used for scatter plots & histograms
import pandas as pd                                        # Pandas - Used for data analysis & manipulation. Used for dataframes(df)
import numpy as np                                         # Numpy - Used for large datasets to perform numerical computing
import sys                                                 # Sys - sys module provides functions and variables used to manipulate different parts of the Python runtime environment
import seaborn as sns                                      # Seaborn - built on top of matplotlib &  used for data visualisation plots such as pairplots
import os                                                  # Operating System - Used for file handling (creation, path management)
import scipy as sp                                         # Scipy - A library for scientific computing, used for statistical regressions


###########################################################################################################


## Task 2: Source & Explore the Iris dataSet

data = skl.datasets.load_iris()                           #Load the iris dataset from scikit-learn

df = pd.DataFrame(data.data, columns=data.feature_names)  # Convert to Pandas Dataframe

print(data)

# Comments: sklearn used for import of iris dataset - covered in Principles of Data Analytics course.
# Reference https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset


###########################################################################################################


## ** Task 3 - Summarise the Data ** Summary Statistics 

# The four variables in the dataset are:
# - sepal length (cm) , sepal width (cm) , petal length (cm) , petal width (cm)
# We will look at the dataframe now & compare vs original data loaded from scikit-learn

print(df.head())                                           # df.head shows us the first 5 rows of the dataset & the variable data that has been collected

print(df.tail())                                           # df.head shows us the last 5 rows of the dataset & the variable data that has been collected

print(df.shape)                                            # df.shape shows us the size of the dataset(rows & columns) that we are working with

print(df.describe())                                       # df.describe() is used to show statistical data about the Iris dataset such as Mean, Minimum, Maximum, Standard Deviation & Median

# Create a file to save the summary

with open('Iris Dataset - Summary Statistics.txt','w') as file:             # Create text file in write mode(r)
    file.write('Iris Dataset - Summary Statistics\n\n')                     # Writes title into created text file. \n to add a new line after created text
    file.write(df.describe().to_string())                                   #.to_string() is used to convert dataframe to string
    file.write("\n\n")                                                      # Add extra(new) lines to seperate from next summary statistic
    
    # Add df.head() to the text file
    file.write("First 5 rows (df.head())\n")
    file.write(df.head().to_string())
    file.write("\n\n")                                                      # Add extra new lines

    # Add df.tail() to the text file
    file.write("Last 5 rows (df.tail())\n")
    file.write(df.tail().to_string())
    file.write("\n\n")                                                      # Add extra new lines

    # Add df.shape to the text file
    file.write(f"Shape of the dataset (df.shape): {df.shape}\n\n")          # Use f string - covered in initial lectures + add extra new lines

    # Add df.describe() to the text file
    file.write("Summary Statistics (df.describe())\n")
    file.write(df.describe().to_string())



# Comments: df head/shape/describe - covered in Principles of Data Analytics course.
# Reference:
# .to_string - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_string.html

# I referred to this youtube video for assistance on writing files  https://www.youtube.com/watch?v=1IYrmTTKOoI


# Addition of extra columns to dataframe

# Add new column 'species' which lables the species numerically ()
# Allows us to analyse the data per species . If this was not added the three flower species would be grouped.

df['species'] = data.target

# Adds new colum 'species_name' with the species names('setosa', 'versicolor', 'virginica). 
# Lamda function matches each value of x to the name from the  data.target_names - see Reference 'chatgpt'

df['species_name'] = df['species'].map(lambda x: data.target_names[x])

#print(df)

# Reference: 
# chatgpt -  Reference From chat gpt -Query: iris species name is not included in my data frame after downloading from scikit-learn, how would I make sure it is included?
#- https://chatgpt.com/share/67ea755a-0d30-800b-ba46-c380efaf35a8


###########################################################################################################







## ** Task 4 -Visualise Features -Histogram
# Saves a histogram of each variable to png files - 

# Review columns in DF
print(df.columns)

# Make sure folder where you want to save the png file exists - Reference os.mkdirs
os.makedirs("Histograms", exist_ok=True)  # Creates the folder if it doesn't exist

#From Data Analyics module - Applicable columns are first 4, dataset will include the added columns of species', 'species_name' so subset needed for 4 feautures
# iloc used to create subset for required features only format is row,columns
df_subset_features=df.iloc[:, 0:4]

# Plot histogram of Features.
axes = df_subset_features.hist(bins =20,alpha = .9 , color='skyblue', edgecolor='black')

# Add axis lables                                           #Note - plt.xlabel/ylabel will only show axis labels for the last plot instead of for each histogram
# plt.xlabel('Value')
# plt.ylabel('Count')

# Flatten the 2D array of axes into a list
axes = axes.flatten()

# Add lables per subplot using for loop
for ax in axes:
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')

# Add spacing so histograms not on top of each other - Reference :plt.tight_layout
plt.tight_layout(pad=2)

# Set the title for all histograms - Reference: Super Title
plt.suptitle("Histogram : Features of Iris Dataset",fontweight='bold')  # Supertitle for all histograms

# Save the histograms to PNG file - see reference : plt.savefig
plt.savefig("Histograms/ Histograms - Iris Variables.png")

# Show the plot
plt.show()

#Reference:
# os.mkdirs - https://docs.python.org/3/library/os.html#os.makedirs
# Super Title - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html#
# plt.savefig - https://www.geeksforgeeks.org/matplotlib-pyplot-savefig-in-python/
# plt.tight_layout - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html









###########################################################################################################




## Task 5 - Investigate & Analyse Relationships - Scatter Plot (i)**

# We will need to amend the df_subset if we are to split the colours per flower species. In earlier itteration the graphs only showed with the features grouped which made it hard to distinguish
# Amended df_subset
df_subset = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species_name']]

# Define the list of features
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

#Create pairplot
sns.pairplot(df_subset, vars=features, hue='species_name')          #vars = points to columns that should be included ; hue = colours the points in species names

# Show the plot
plt.show()

# Reference - 
# https://campus.datacamp.com/courses/exploratory-data-analysis-in-python/relationships-in-data?ex=5 ,
# https://campus.datacamp.com/courses/understanding-data-visualization/the-color-and-the-shape?ex=8
# https://seaborn.pydata.org/generated/seaborn.pairplot.html


###########################################################################################################


## 5.  **Investigate *& Analyse Relationships - Scatter Plot (ii)**

#Re-check column names
print(df.columns)

# Assign x and y variable
x = df['petal length (cm)']
y = df['petal width (cm)']


# Create a figure and an axis
fig , ax = plt.subplots()

# Scatter plot
#sc_plot = ax.scatter(x,y,marker ='8', c=df['species']) # 'Species' used as numerical input required
sc_plot = ax.scatter(x,y,marker ='8', c=df['species'].astype('category').cat.codes, cmap='viridis') # 'Species' used as numerical input required & # Mapping species to numerical values

# Labels
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')

# Add title
plt.title('Scatter Plot/Polyfit - Petal Length and Width for Iris Species')

# Reference from chatgpt for adding legend
#I need to add a legend per species to this code- what is  the best way to do this?
#https://chatgpt.com/share/680e41e5-a4f4-800b-9004-faa642337b82


# Now use the 'species_name' column for legend labels
species_labels = df['species_name'].unique()
codes = df['species'].astype('category').cat.codes.unique()

# Match species names to their correct colors
colors = sc_plot.cmap(sc_plot.norm(codes))

handles = []
for species, color in zip(species_labels, colors):
    handle = plt.Line2D(
        [], [], marker='8', color='w', markerfacecolor=color,
        label=species, markersize=10
    )
    handles.append(handle)


# Use polyfit to fit a line to the data
m, c = np.polyfit(x,y,1)

# Plot a line using polyfit & assign to variable                            # line_handle, - , is required so full list from ax.plot is not returned
line_handle, = ax.plot(x ,m * x + c, color='red',label='Fit Line')

# Add regression line to handles list
handles.append(line_handle)

# Add the legend
ax.legend(handles=handles, title="Species Name & Regression Line")

# Show plot
plt.show()







###########################################################################################################






## Task 6 - Other Analysis - Linear Regression**


# x & y values that are required
x = df['petal length (cm)']
y = df['petal width (cm)']

 # Use scipy to fit a line.   # Reference - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
sp.stats.linregress(x,y)

#We use sp.stats.linregress to calculate:
#1. Slope and Intercept of best fit line or regression line
#2. Get the r value - the correlation of values ( We will need this to calculate the coefficient of determination r² )
#3. Get the p-value - statistial signifance of relationship
#4. Get standard error  - how values deviates from best fit line

# Assign line to variable 'fit'
fit = sp.stats.linregress(x,y)

# R2 value - Measures variance in the dependant variable(petal width)
r_squared=fit.rvalue**2

print(f"The R² value coefficient shows a value of {r_squared * 100}%")


###########################################################################################################

## Task 7: Analyse Class Distributions - Boxplot

# Create figure, axis.
fig, ax = plt.subplots()


#Target data - ['setosa', 'versicolor', 'virginica'

# Need petal length per species of flower -Ref for how to create subset Ref iloc- https://youtube.com/watch?v=xvpNA7bC8cs&si=Ds7TxZjxR8qzDDgB
#Add explanation of loc function + format of code.


setosa_data = df.loc[df.species_name == 'setosa','petal length (cm)']
versicolor_data = df.loc[df.species_name == 'versicolor','petal length (cm)']
virginica_data = df.loc[df.species_name == 'virginica','petal length (cm)']


#Check output
#print(setosa_data)
#print(versicolor_data)
#print(virginica_data)

# Create a boxplot
ax.boxplot([setosa_data,versicolor_data,virginica_data])

# Label x Axis Names
ax.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'])

# Add title
plt.title('Box plot - Petal Length (cm)')




###########################################################################################################


# 8.  **Other Analysis - Compute Correlations** Correlation Heatmap


#Applicable columns are first 4, dataset will include the added columns of species', 'species_name' so subset needed.
#iloc used similar to above for aplicabale columsn only
df_subset=df.iloc[:, 0:4]
#print(df_subset)


# Correlation Coefficient . Numeric Only so 
corr_matrix = df_subset.corr()         # Reference : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

print(corr_matrix)


# heatmap using matplotlib

# Create figure, axis.
fig, ax = plt.subplots()

# Plot heatmap
image = ax.imshow(corr_matrix,cmap='coolwarm')

# Add colorbar
fig.colorbar(image, ax=ax, label='Correlation Coefficient')

# Labels

ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.index)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_matrix.index)

#Add title
plt.title('Correlation Heatmap')

# Plot
plt.tight_layout()
plt.show()

























