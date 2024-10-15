#importing necessary libraries
import pandas as pd
from MLPipeline.TrainModel import TrainModel
from MLPipeline.Preprocessing import Preprocessing

# Reading the data
df = pd.read_csv("Input/data.csv")

#put name of columns you want to drop
df = Preprocessing(df).drop(["customer_id", "phone_no", "year"]) #comment this line if there are no columns to drop

#dropping null values
data =Preprocessing(df).dropna()

#scaling numerical features
data=Preprocessing(data).scale()

#label encoding categorical features
data=Preprocessing(data).encode()

# splitting data into train and test
target_col='no_of_days_subscribed' #Put target column name here
X_train, X_test, y_train, y_test = Preprocessing(data).split_data(target_col)

#converting data into tensor form
n_features,X_train, y_train, X_test, y_test = Preprocessing(data).convert_to_tensor(X_train,y_train,X_test,y_test)


# # Training the network
TrainModel(n_features, X_train, y_train, X_test, y_test)

