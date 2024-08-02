import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline


# Import the dataset.

raw_immo_df = pd.read_json('final_dataset.json')


# Data validation adn data cleaning.

dictionary_types = {'Url': 'string', 'BathroomCount': 'Int64', 'BedroomCount': 'Int64', 'ConstructionYear': 'Int64',
                    'Country': 'string', 'District': 'string', 'Fireplace': 'boolean', 'FloodingZone': 'category', 'Furnished': 'boolean',
                    'Garden': 'boolean', 'GardenArea': 'Int64', 'Kitchen': 'category', 'LivingArea': 'Int64', 'Locality': 'string',
                    'MonthlyCharges': 'Float64', 'NumberOfFacades': 'Int64', 'PEB': 'category', 'PostalCode': 'string', 'Price': 'Float64',
                    'PropertyId': 'string', 'Province': 'string', 'Region': 'string', 'RoomCount': 'Int64', 'ShowerCount': 'Int64',
                    'StateOfBuilding': 'category', 'SubtypeOfProperty': 'category', 'SurfaceOfPlot': 'Int64', 'SwimmingPool': 'boolean',
                    'Terrace': 'boolean', 'ToiletCount': 'Int64', 'TypeOfProperty': 'category', 'TypeOfSale': 'category'
                    }
raw_immo_df = raw_immo_df.astype(dictionary_types)

numerical_columns = raw_immo_df.select_dtypes(['Int64', 'Float64']).columns
boolean_columns = raw_immo_df.select_dtypes(['boolean']).columns
string_columns = raw_immo_df.select_dtypes(['string']).columns
category_columns = raw_immo_df.select_dtypes(['category']).columns


raw_immo_df.dropna(subset=["District", "Locality", "Province", "Region"], inplace=True)

raw_immo_df["Locality"] = raw_immo_df["Locality"].str.capitalize()

raw_immo_df["FloodingZone"] = raw_immo_df["FloodingZone"].apply(
    lambda x: x.replace('_N_', '_AND_').replace('_', ' ').capitalize())

raw_immo_df["Kitchen"] = raw_immo_df["Kitchen"].apply(
    lambda x: x.replace('USA_', '').replace('_', ' ').capitalize())

raw_immo_df.loc[raw_immo_df["Kitchen"] == "Not installed", "Kitchen"] = "Uninstalled"
raw_immo_df = raw_immo_df.astype({"Kitchen": "category"})

raw_immo_df["PEB"] = raw_immo_df["PEB"].apply(lambda x: x.split('_')[0])
raw_immo_df = raw_immo_df.astype({"PEB": "category"})

raw_immo_df["StateOfBuilding"] = raw_immo_df["StateOfBuilding"].apply(
    lambda x: x.replace('_', ' ').capitalize())

raw_immo_df["SubtypeOfProperty"] = raw_immo_df["SubtypeOfProperty"].apply(
    lambda x: x.replace('_', ' ').capitalize())

raw_immo_df["TypeOfProperty"] = raw_immo_df["TypeOfProperty"].apply(
    (lambda x: 'House' if x == 1 else 'Apartment'))

raw_immo_df["TypeOfSale"] = raw_immo_df["TypeOfSale"].apply(
    lambda x: x.replace('_', ' ').capitalize())


# In the dataset, there are 104942 'Residential sale' properties , 13450 'Residential monthly rent' properties
# and only 315 properties with other type of sale. Let's keep only the 'Residential sle'.
 
residential_sale_df = raw_immo_df[raw_immo_df["TypeOfSale"] == 'Residential sale']

# There are only 97 properties with a price less that 45000. Let's drop these.

residential_sale_df = residential_sale_df[residential_sale_df["Price"] > 45000]



# Adressing missing data.


final_df = residential_sale_df.fillna({"Fireplace": False,
                                       "Furnished": False,
                                       "Garden": False,
                                       "SwimmingPool": False,
                                       "Terrace": False,
                                       "GardenArea": 0})

final_df = final_df.fillna({"FloodingZone": final_df["FloodingZone"].mode().iloc[0]})

final_df = final_df.fillna(
    {"Kitchen": final_df["Kitchen"].mode().iloc[0],
     "PEB": final_df["PEB"].mode().iloc[0],
     "StateOfBuilding": final_df["StateOfBuilding"].mode().iloc[0]
     })


# A cleaned DataFrame for the model.


columns_names = ["Furnished", "Garden", "SwimmingPool", "Terrace",
                 "ConstructionYear", "BedroomCount", "BathroomCount", "LivingArea", "GardenArea", "SurfaceOfPlot",
                 "PEB", "StateOfBuilding", "Kitchen", "District", "TypeOfProperty", "SubtypeOfProperty",
                 "Price"]

df = final_df[columns_names]

df = df.dropna(subset=["ConstructionYear", "BathroomCount", "LivingArea"])

# The surface of plot is missing for all appartment. Let's fill them with the living area.

df["SurfaceOfPlot"] = df["SurfaceOfPlot"].fillna(df.LivingArea)



# Feature engineering


model_features = ["Furnished", "Garden", "SwimmingPool", "Terrace",
                  "ConstructionYear", "BedroomCount", "BathroomCount", "LivingArea", "GardenArea", "SurfaceOfPlot",
                  "PEB", "StateOfBuilding", "Kitchen", "District", "TypeOfProperty", "SubtypeOfProperty"]

features = df[model_features]


cat_order = [["G", "F", "E", "D", "C", "B", "A", "A+", "A++"],
             ["To restore", "To renovate", "To be done up",
                 "Just renovated", "Good", "As new"],
             ["Uninstalled", "Installed", "Semi equipped", "Hyper equipped"]]

pre = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), ["Furnished", "Garden", "SwimmingPool", "Terrace",
                                 "ConstructionYear", "BedroomCount", "BathroomCount", "LivingArea", "GardenArea", "SurfaceOfPlot"]),
        ("cat", OneHotEncoder(drop="first", sparse_output=False),
         ["District", "TypeOfProperty", "SubtypeOfProperty"]),
        ("ord", OrdinalEncoder(categories=cat_order),
         ["PEB", "StateOfBuilding", "Kitchen"])
    ]
)

X = pre.fit_transform(features)

y = df["Price"].values


# Store the tuple (df, pre, X, y) as pickle file, where df is the cleaned DataFrame,
# X is the features numpy array, y the target, # and pre the ColumnTransformer instance.

with open("features_target.pkl", "wb") as f:
    pickle.dump((df, pre, X, y), f, protocol=5)


# An alternative DataFrame without outliers.

iqr = df["Price"].quantile(0.75) - df["Price"].quantile(0.25)
lower_threshold = df["Price"].quantile(0.25) - (1.5 * iqr)
upper_threshold = df["Price"].quantile(0.75) + (1.5 * iqr)

drop_outliers = np.logical_and(
    df["Price"] > lower_threshold, df["Price"] < upper_threshold)

without_outliers_df = df[drop_outliers]

features_wo = without_outliers_df[model_features]
X_wo = pre.fit_transform(features_wo)
y_wo = without_outliers_df["Price"].values


# Store the tuple (without_outliers_df, X, y) as pickle file,
# where without_outliers_df is the cleaned DataFrame without outliers,
# X_wo is the features numpy array and y_wo the target.

with open("features_target_wo.pkl", "wb") as f:
    pickle.dump((without_outliers_df, pre, X_wo, y_wo), f, protocol=5)


print(f"""
      Done!
      The cleaned DataFrame with shape {df.shape},
      the features, and the target have been stored.
      
      Another DataFrame without the outliers, with shape {without_outliers_df.shape},
      the corresponding features and target have been also stored.
      """)