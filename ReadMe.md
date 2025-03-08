# Exploratory Data Analysis (EDA)
This repository contains code to perform Principal Component Analysis (PCA) and Factor Analysis on sleepiness questionnaire data, along with a reliability assessment using Cronbach's alpha. The data is analyzed to uncover the underlying factors of subjective daytime sleepiness and to examine how these factors vary with demographic and sleep-related variables.

![alt text](presentation\correlation_ess_sss.png "Correlation Between Questionnaires")


## Definition of the Sleep Scale Questionnaires
### Situational Sleepiness Scale (SSS)
The Situational Sleepiness Scale (SSS) assesses the likelihood of falling asleep in various situations. The score is calculated by summing the responses to several questions, with a cut-off score of 10 to indicate a sleepiness problem.

### Epworth Sleepiness Scale (ESS)
The Epworth Sleepiness Scale (ESS) assesses the general level of daytime sleepiness. The score is calculated by summing the responses to several questions, with a cut-off score of 10 to indicate a sleepiness problem.

# Analysis
PCA and Factor Analysis
## Principal Component Analysis (PCA)
PCA is used to reduce the dimensionality of the data and identify the principal components that explain the most variance in the dataset. The steps include:

1. Standardize the data to have a mean of 0 and a standard deviation of 1.
2. Fit PCA to the standardized data and transform the data into principal components.
3. Visualize the results to interpret the principal components

## Factor Analysis
Factor Analysis is used to identify the underlying factors that explain the observed correlations among variables. The steps include:

1. Standardize the data to ensure comparability.
2. Perform Factor Analysis with a specified number of factors and rotation (e.g., varimax).
3. Calculate Cronbach's alpha for each identified factor to assess reliability.

## Cronbach's Alpha
Cronbach's alpha is a measure of internal consistency or reliability of a set of items. It is calculated using the variance of each item and the total variance of the sum of items.

## Explanation File Structure 
The src folder contains the course code of the: 
1. pre-processing pipeline:
   - pre_process_first_cross_sectional.py
   - pre_process_second_cross_sectional.py
   - pre_process_merge_cross_sectional.py
2. Correlation and PCA analysis (statistical_analysis.py)
3. Factorial Analysis (subject_daytime_analysis.py)
4. Table creation to present the demographics and responses (table_dem_questions.py)

The config folder contains config.py where the directories and main configurations are defined. 

The questionnaires library contains two classes, one for each sleepiness scale questionnaire where 
the definition in a python-structure format are present. 

