# Credit Card Approval
# Table of Content
- [Demo](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#demo)
- [Problem Statement](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#problem-statement)
- [Attribute](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#attribute)
- [Data Description](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#data-description)
- [Data Validation](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#data-validation)
- [Data Insertion in Database](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#data-insertion-in-database)
- [Model Training](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#model-training)
- [Prediction Data Description](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#prediction-data-description)
- [Data Validation](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#data-validation-1)
- [Data Insertion in Database](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#data-insertion-in-database-1)
- [Prediction](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#prediction)
- [Deployment](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#deployment)
- [Team member](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#team-member)
- [Credit](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#credit)
- [Dataset](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#dataset)
- [Mentors](https://github.com/GokulDev4U/Credit_Approval/blob/master/README.md#mentors)


# Demo
  https://credit-approved.herokuapp.com/
  
# Problem Statement
To build a classification methodology to predict the Credit Approved or not based on the given training data. 
# Attribute 
This file concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

This dataset is interesting because there is a good mix of attributes -- continuous, nominal with small numbers of values, and nominal with larger numbers of values. There are also a few missing values.

- A1: b, a.
- A2: continuous.
- A3: continuous.
- A4: u, y, l, t.
- A5: g, p, gg.
- A6: c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
- A7: v, h, bb, j, n, z, dd, ff, o.
- A8: continuous.
- A9: t, f.
- A10: t, f.
- A11: continuous.
- A12: t, f.
- A13: g, p, s.
- A14: continuous.
- A15: continuous.
- A16: +,- (class attribute)

# Data Description
The client will send data in multiple sets of files in batches at a given location. Data will contain different classes of Credit Approval and 15 columns of different values.
"Class" column will have two unique values “+’’ & “-”

Apart from training files, we also require a "schema" file from the client, which contains all the relevant information about the training files such as:
 Number of Columns, Name of the Columns, and their datatype.
 
# Data Validation 
In this step, we perform different sets of validation on the given set of training files.  

1. Number of Columns - We validate the number of columns present in the files, and if it doesn't match with the value given in the schema file, then the file is moved to "Rejected folder" else moved to “Validate folder”
For training: training_data_reject,training_data_validate
For predicting: predicting_data_reject,predicting_data_validate

2. Name of Columns - The name of the columns is validated and should be the same as given in the schema file. If not, then the file is moved to "Rejected folder".

3. The datatype of columns - The datatype of columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is moved to "Rejected folder".

4.Null values in columns - If any of the columns in a file have all the values as NULL or missing, we discard such a file and move it to "Rejected folder".



# Data Insertion in Database
 
1) Database Creation and connection - Create a database with the given name passed. If the database is already created, open the connection to the database. 
2) Table creation in the database - Table with name - "Train Data", is created in the database for inserting the files in the "Validate Folder" based on given column names and datatype in the schema file. If the table is already present, then the new table is not created and new files are inserted in the already present table as we want training to be done on new as well as old training files.     
3) Insertion of files in the table - All the files in the "Validate Folder" are inserted in the above-created table. If any file has invalid data type in any of the columns, the file is not loaded in the table and is moved to "Rejected Folder".
 
# Model Training 
1) Data Export from Db - The data in a stored database is exported as a CSV file to be used for model training.
2) Data Preprocessing   
   -  Drop columns not useful for training the model. Such columns were selected while doing the EDA.
   -  Replace the invalid values(‘?’) with numpy “nan” so we can use imputer on such values.
   -  Encode the categorical values
   -  Check for null values in the columns. If present, impute the null values using the KNN imputer.
   -  top four feature is selected with Selectkbest & chi2
3) Model Selection - After feature selection, we find the best model for each cluster. We are using two algorithms, "Random Forest" and "Xgboost". For each cluster, both the algorithms are passed with the best parameters derived from GridSearch. We calculate the AUC scores for both models and select the model with the best score. Similarly, the model is selected for each cluster. All the models for every cluster are saved for use in prediction. 
 
# Prediction Data Description
 The client will send data in multiple sets of files in batches at a given location. Data will contain different classes of Credit Approval and 15 columns of different values.
Apart from prediction files, we also require a "schema" file from the client, which contains all the relevant information about the training files such as:
 Number of Columns, Name of the Columns, and their datatype.

# Data Validation  
In this step, we perform different sets of validation on the given set of training files.  
1.Number of Columns - We validate the number of columns present in the files, and if it doesn't match with the value given in the schema file, then the file is moved to "Rejected folder" else moved to “Validate folder”
For training: training_data_reject,training_data_validate
For predicting: predicting_data_reject,predicting_data_validate

2.Name of Columns - The name of the columns is validated and should be the same as given in the schema file. If not, then the file is moved to "Rejected folder".

3.The datatype of columns - The datatype of columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is moved to "Rejected folder".

4. Null values in columns - If any of the columns in a file have all the values  as NULL or missing, we discard such a file and move it to "Rejected folder". 

# Data Insertion in Database 

1) Database Creation and connection - Create a database with the given name passed. If the database is already created, open the connection to the database. 
2) Table creation in the database - Table with name - "Predict Data", is created in the database for inserting the files in the "Validate Folder" based on given column names and datatype in the schema file. If the table is already present, then the new table is not created and new files are inserted in the already present table as we want training to be done on new as well as old training files.     
3) Insertion of files in the table - All the files in the "Validate Folder" are inserted in the above-created table. If any file has invalid data type in any of the columns, the file is not loaded in the table and is moved to "Rejected Folder".


# Prediction 
 
1) Data Export from Db - The data in the stored database is exported as a CSV file to be used for prediction.
2) Data Preprocessing   
   - Drop columns not useful for training the model. Such columns were selected while doing the EDA.
   - Replace the invalid values with numpy “nan” so we can use imputer on such values.
   - Encode the categorical values
   - Check for null values in the columns. If present, impute the null values using the KNN imputer.
   - top four feature is selected with Selectkbest & chi2
3) Prediction -  the best model is loaded and is used to predict the data .
4) Once the prediction is made, the predictions along with the original names before label encoder are saved in a CSV file at a given location and the location is returned to the client.

# Deployment
Go to https://cloud.google.com/ and create an account if already haven’t created one. 
Then go to the console of your account.
- Go to IAM and admin(highlighted) and click manage resources. 
- Click CREATE PROJECT to create a new project for deployment.
- Once the project gets created, select App Engine and select Dashboard.
- Go to https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe to download the google cloud SDK to your machine.
- Click Start Tutorial on the screen and select Python app and click start.
- Check whether the correct project name is displayed and then click next.
- Create a file ‘app.yaml’ and put ‘runtime: python37’ in that file.
- Create a ‘requirements.txt’ file by opening the command prompt/anaconda prompt, navigate to the project folder and enter the command ‘pip freeze > requirements.txt’.
  It is recommended to use separate environments for different projects.
- Your python application file should be called ‘main.py’. It is a GCP specific requirement.
- Open command prompt window, navigate to the project folder and enter the command gcloud init to initialise the gcloud context.
- It asks you to select from the list of available projects.
- Once the project name is selected, enter the command gcloud app deploy app.yaml --project <project name>.
- After executing the above command, GCP will ask you to enter the region for your application. Choose the appropriate one.
- GCP will ask for the services to be deployed. Enter ‘y’ to deploy the services.
- And then it will give you the link for your app, and the deployed app looks like:
- To save money, go to settings and disable your app.
  
# Team member
- Gokul Pisharody(myself)
- Sanjay Kumar
- Devi  Arumugam
  
# credit
  http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html

# Dataset
  https://archive.ics.uci.edu/ml/datasets/Credit+Approval
  
# Mentors
- virat sir
- Mohit sir



