import streamlit as st

import numpy as np
import pandas as pd

import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PassiveAggressiveRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, median_absolute_error
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV, GridSearchCV


model_features = ["Furnished", "Garden", "SwimmingPool", "Terrace",
                  "ConstructionYear", "BedroomCount", "BathroomCount", "LivingArea", "GardenArea", "SurfaceOfPlot",
                  "PEB", "StateOfBuilding", "Kitchen", "District", "TypeOfProperty", "SubtypeOfProperty"]

names_of_districts = sorted(('Brugge', 'Tournai', 'Veurne', 'Hasselt', 'Brussels', 'Nivelles',
                             'Mechelen', 'Halle-Vilvoorde', 'Sint-Niklaas', 'Oostend', 'Antwerp', 'Ieper',
                             'Mons', 'Namur', 'Philippeville', 'Soignies', 'Leuven', 'Charleroi',
                             'Liège', 'Maaseik', 'Verviers', 'Aalst', 'Tongeren', 'Marche-en-Famenne',
                             'Kortrijk', 'Gent', 'Eeklo', 'Diksmuide', 'Dendermonde', 'Waremme',
                             'Huy', 'Oudenaarde', 'Dinant', 'Neufchâteau', 'Mouscron', 'Tielt', 'Roeselare',
                             'Turnhout', 'Thuin', 'Arlon', 'Virton', 'Ath', 'Bastogne'))

names_of_types = ('Apartment', 'House')

names_of_subtypes = ('Apartment', 'Apartment block', 'Bungalow', 'Castle', 'Chalet', 'Country cottage',
                     'Duplex', 'Exceptional property', 'Farmhouse', 'Flat studio', 'Ground floor', 'House',
                     'Kot', 'Loft', 'Manor house', 'Mansion', 'Mixed use building', 'Other property',
                     'Pavilion', 'Penthouse', 'Service flat', 'Show house', 'Town house', 'Triplex', 'Villa')


def to_boolean(value: str) -> bool:
    if value == "No":
        return False
    elif value == "Yes":
        return True
    else:
        raise ValueError("The argument must be 'Yes' or 'No'.")


st.title("Immo-Eliza Application")
st.header(":blue[Property Price Estimator]", divider="blue")

top_col_1, top_col_2 = st.columns(2)

with top_col_1:
    st.image("images/real_estate_agent.png")

with top_col_2:
    st.image("images/euro.png")

st.markdown(
    """
    This web application allows you to estimate the price of any real estate located in Belgium. 
    Fill out the form and click on the button at the bottom of the form to view the estimate.
    """
)


st.subheader(":blue[Property's features]")


with st.form("Property features"):
    middle_col_1, middle_col_2 = st.columns(2)

    with middle_col_1:
        type_of_property = st.selectbox(
            "Type of property:", names_of_types, key="type_of_property")

        subtype_of_property = st.selectbox(
            "Subtype of property:", names_of_subtypes, key="subtype_of_property")

        construction_year = st.number_input("Construction Year:", key="construction_year",
                                            min_value=1900,
                                            step=1)

        furnished = st.radio("Is the property furnished?", [
                             "No", "Yes"], key="furnished")

        garden = st.radio("Does the property have a garden?",
                          ["No", "Yes"], key="garden")

        garden_area = st.number_input(
            "Garden area in squared meter:", key="garden_area", min_value=0, step=1)

        swimming_pool = st.radio("Does the property have a swimming pool?",
                                 ["No", "Yes"], key="swimming_pool")

        terrace = st.radio("Does the property have a terrace?",
                           ["No", "Yes"], key="terrace")

    with middle_col_2:
        district = st.selectbox(
            "District:", names_of_districts, key="district")

        bedroom_count = st.number_input("Number of bedrooms:", key="bedroom_count",
                                        min_value=0,
                                        step=1)

        bathroom_count = st.number_input("Number of bathrooms:", key="bathroom_count",
                                         min_value=0,
                                         step=1)

        living_area = st.number_input(
            "Living area in squared meter:", key="living_area", min_value=0, step=1)

        surface_of_plot = st.number_input("Surface of plot in squared meter:",
                                          key="surface_of_plot", min_value=0, step=1)

        peb = st.selectbox("House Energy Rating (PEB):",
                           ("G", "F", "E", "D", "C", "B", "A", "A+", "A++"),
                           key="peb")

        state_of_building = st.selectbox("State of the building:",
                                         ("To restore", "To renovate", "To be done up",
                                          "Just renovated", "Good", "As new"),
                                         key="state_of_building")

        kitchen = st.selectbox("State of the kitchen:",
                               ("Uninstalled", "Installed",
                                "Semi equipped", "Hyper equipped"),
                               key="kitchen")

    submit = st.form_submit_button("Estimate the price", type="primary")


features_list = [furnished, garden, swimming_pool, terrace, construction_year, bedroom_count,
                 bathroom_count, living_area, garden_area, surface_of_plot,
                 peb, state_of_building, kitchen, district, type_of_property, subtype_of_property]

# ["Furnished", "Garden", "SwimmingPool", "Terrace", "ConstructionYear", "BedroomCount",
# "BathroomCount", "LivingArea", "GardenArea", "SurfaceOfPlot",
# "PEB", "StateOfBuilding", "Kitchen", "District", "TypeOfProperty", "SubtypeOfProperty"]

property_dictionary = {"Furnished": [to_boolean(st.session_state.furnished)],
                       "Garden": [to_boolean(st.session_state.garden)],
                       "SwimmingPool": [to_boolean(st.session_state.swimming_pool)],
                       "Terrace": [to_boolean(st.session_state.terrace)],
                       "ConstructionYear": [st.session_state.construction_year],
                       "BedroomCount": [st.session_state.bedroom_count],
                       "BathroomCount": [st.session_state.bathroom_count],
                       "LivingArea": [st.session_state.living_area],
                       "GardenArea": [st.session_state.garden_area],
                       "SurfaceOfPlot": [st.session_state.surface_of_plot],
                       "PEB": [st.session_state.peb],
                       "StateOfBuilding": [st.session_state.state_of_building],
                       "Kitchen": [st.session_state.kitchen],
                       "District": [st.session_state.district],
                       "TypeOfProperty": [st.session_state.type_of_property],
                       "SubtypeOfProperty": [st.session_state.subtype_of_property]}

dictionary_types = {'Furnished': 'boolean',
                    'Garden': 'boolean',
                    'SwimmingPool': 'boolean',
                    'Terrace': 'boolean',
                    'ConstructionYear': 'Int64',
                    'BedroomCount': 'Int64',
                    'BathroomCount': 'Int64',
                    'LivingArea': 'Int64',
                    'GardenArea': 'Int64',
                    'SurfaceOfPlot': 'Int64',
                    'PEB': 'category',
                    'StateOfBuilding': 'category',
                    'Kitchen': 'category',
                    'District': 'string',
                    'TypeOfProperty': 'category',
                    'SubtypeOfProperty': 'category'}

df_property = pd.DataFrame(property_dictionary)

df_property = df_property.astype(dictionary_types)


with open("models/features_target_wo.pkl", "rb") as f:
    df, pre, X, y = pickle.load(f)

with open("models/predictor_hgbr.pkl", "rb") as f:
    hgb_regressor, hgbr_metrics = pickle.load(f)

with open("models/predictor_xgbr.pkl", "rb") as f:
    xgb_regressor, xgbr_metrics = pickle.load(f)

with open("models/predictor_rfr.pkl", "rb") as f:
    rf_regressor, rfr_metrics = pickle.load(f)


models = [(hgb_regressor, hgbr_metrics), (xgb_regressor, xgbr_metrics), (rf_regressor, rfr_metrics)]

model, metrics = models[1]


X_property = pre.transform(df_property)


prediction = int(model.predict(X_property)[0])


if submit:
    bottom_col_1, bottom_col_2 = st.columns(2)

    with bottom_col_1:
        st.subheader(":blue[Estimated Price:]")

    with bottom_col_2:
        st.subheader(f"{max(0, prediction)} €")

with st.sidebar:
    st.title(":red[Technical Information]")

    st.markdown(
        f"""
        To design this application, we used a supervised learning model,
        called **:blue[{metrics["class"]}]**, which we trained on 80% of
        a dataset consisting of 56307 entries.

        Here are the performance indicators of the model.

        R2-score (train): {round(metrics["R2-score (train)"], 2)}

        R2-score (test): {round(metrics["R2-score (test)"], 2)}

        95% Confindence interval of the score: {np.round(metrics["95% Confindence interval of the score"], 2)}

        Mean Absolute Error: **:blue[{int(metrics["Mean Absolute Error"])} €]**

        Root Mean Squared Error: {int(metrics["Root Mean Squared Error"])} €

        Median Absolute Error: {int(metrics["Median Absolute Error"])} €
        """
    )
