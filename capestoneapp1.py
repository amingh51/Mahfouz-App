# importing libraries 

import pandas as pd 
import numpy as np 
import re 
import streamlit as st 
import hydralit_components as hc
import requests
from streamlit_lottie import st_lottie
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px  
import streamlit_card as st_card
import numpy as np
from pathlib import Path
import base64
from datetime import datetime
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV



    # Data processing Code 

    # importing Data 
########################################################################################################################################################################
########################################################################################################################################################################
    # Defining data processing function
def process_data(product_sales_data):
        
        dollar = pd.read_csv("usd-lbp.csv")

        file_path = product_sales_data
        # Read product sales data
        product_sales = pd.read_csv(file_path, encoding='latin1')

        ###fixing dollar rate data 
        # Convert 'DateTime' column to datetime format
        dollar['DateTime'] = pd.to_datetime(dollar['DateTime'])

        # Filter out data before 2021
        dollar = dollar[dollar['DateTime'].dt.year >= 2021]

        # Calculate the average exchange rate for each day
        daily_average = dollar.groupby(dollar['DateTime'].dt.date)['USD to LBP'].mean().reset_index()

        df_daverage = pd.DataFrame(daily_average)

        ###Adding dollar prices to Mahfouz Data 
        # Convert 'DateTime' column to datetime format
        daily_average['DateTime'] = pd.to_datetime(daily_average['DateTime'])

        # Convert the 'Date' column to datetime type
        product_sales['Date'] = pd.to_datetime(product_sales['Date'])

        # Merge 'product_sales' with 'daily_average' based on the 'Date' column
        product_sales_dollar = pd.merge(product_sales, daily_average, left_on='Date', right_on='DateTime', how='left')

        # Calculate Selling price and Cost price in USD based on exchange rates
        product_sales_dollar['Selling price (USD)'] = product_sales_dollar['Selling price'] / product_sales_dollar['USD to LBP']
        product_sales_dollar['Cost price (USD)'] = product_sales_dollar['Cost price'] / product_sales_dollar['USD to LBP']

        # Drop DateTime and USD to LBP columns
        product_sales_dollar.drop(['DateTime', 'USD to LBP'], axis=1, inplace=True)

        ### Adding new columns/categorization = feature engineering

        # Dropping LBP columns
        #New DataFrame 'df1' by dropping specified columns
        columns_to_drop = ['Selling price', 'Cost price']
        product_sales_dollar = product_sales_dollar.drop(columns=columns_to_drop)

        ## Age group Categorization
        def categorize_age_group(row):
            barcode = row['Barcode']
            product_name = row['Product name']

            if re.match('^b', barcode, flags=re.IGNORECASE):
                return 'Baby'
            elif re.match('^[cg]', barcode, flags=re.IGNORECASE):
                return 'Child'
            elif re.match('^[mw]', barcode, flags=re.IGNORECASE):
                return 'Adult'
            elif re.match('^l', barcode, flags=re.IGNORECASE):
                product_lower = product_name.lower()
                if 'khadam' in product_lower:
                    return 'Adult'
                elif 'baby' in product_lower:
                    return 'Baby'
                elif 'child' in product_lower:
                    return 'Child'
                else:
                    return 'Others'
            else:
                return 'Others'

        product_sales_dollar['Age Group'] = product_sales_dollar.apply(categorize_age_group, axis=1)

        ## porduct gender categorization
        def categorize_gender(Product_name):
            if re.search(r'\b(girl|lady|ladies|wmn|women|female|khadam|collant|child girl|girl child|fizou|vizou|fuseau|girls|wmns|womens|females|birl|wm)\b', Product_name, re.IGNORECASE):
                return "Female"
            elif re.search(r'\b(men|man|male|guy|boys|mens|mans|child|guys|boy|mwn)\b', Product_name, re.IGNORECASE):
                return "Male"
            elif re.search(r'\b(baby|bb)\b', Product_name, re.IGNORECASE):
                return "Un-identified"
            else:
                return "Un-identified"


        # Apply the function to create a new column 'Gender'
        product_sales_dollar['Gender'] = product_sales_dollar['Product name'].apply(categorize_gender)


        ##Item Type Categorization

        # Define a function to categorize the product names
        def categorize_product(product_name):
            # Mapping of categories to respective keywords (all in lowercase)
            categories = {
                    'Belt': ['belt', 'belts', 'zennar','zenar','zinnar'],
                    'Blouse': ['blouse', 'bloue', 'bloues','blousr'],
                    'T-shirt': ['t-shirt', 'tshirt', 'tchirt', 'tchrt'],
                    'Sweater': ['sweater', 'sweeter', 'sweatcher', 'pullover','kim tawil','kim taweel'],
                    'Hoodie': ['hoodie', 'hoodies', 'hoody'],
                    'Boxer': ['boxer', 'boxers', 'bxr', 'brief','breif'],
                    'Slip Underwear': ['underwear', 'kilot', 'string', 'sleep', 'slip', 'slips', 'sleeps', 'under'],
                    'Bra': ['bra', 'bras', 'soutian', 'soutien','sotian', 'brassires','soyien','sotien','soutin','soutein'],
                    'Jacket': ['jacket', 'jaket', 'jacet', 'jac'],
                    'Short': ['short', 'shorts','shourt','chort','chourt'],
                    'Dress': ['dress', 'drss'],
                    'Gillet': ['gilett', 'gillet', 'jillet', 'jilet', 'gilet', 'jilit','jiliet','jillit','jiliet','jiliiet'],
                    'Hijab': ['isharb', 'thjeb'],
                    'Pants': ['pnt', 'pant', 'pantalon', 'pantalon', 'pants'],
                    'Robe': ['bornos', 'bournos', 'bornous', 'bournous', 'robe'],
                    'Jeans': ['jens', 'jean', 'jeans', 'jins', 'jns', 'jeanes','jaens'],
                    'Trousers': ['trouser', 'trousers', 'trousier'],
                    'Vest': ['vest', 'vests', 'bretel', 'brettel', 'brutel', 'brotel', 'broutel'],
                    'Pyjamas': ['pyjama', 'pyjamas', 'pygama', 'pygamas', 'pyjam', 'pygam', 'pygamy', 'pygami', 'pjm','pyagama','pyajama'],
                    'Shoes': ['shoe', 'shoes'],
                    'Slippers': ['slipper', 'slippers', 'sliper', 'slipers', 'sleeper'],
                    'Socks': ['sock', 'socks', 'sox', 'socs', 'soc', 'sok', 'soks'],
                    'Collant': ['collant', 'collants', 'colant', 'colants', 'collont', 'colont', 'collonts', 'colonts','collon','collan'],
                    'Leggings': ['legging', 'leggings', 'fizo', 'fizou', 'fuseau', 'vizou'],
                    'Set': ['set', 'sets','blouse & pants'],
                    'Beach item': ['beach'],
                    'Swimwear': ['mayo', 'swim', 'swimming','mayyo','swimming short','swimming shorts'],
                    'Cap': ['cap', 'kab', 'kap'],
                    'Hat': ['hat'],
                    'Gloves': ['kaf', 'glove', 'gloves'],
                    'Cap + Scarf': ['cap with shall', 'cap with chall', 'cap with shal', 'cap with chal',' cap + shall', 'cap + chall'],
                    'Cap + Gloves': ['cap + glove', 'cap with kaf','cap with gloves', 'cap + kaf'],
                    'Bolero': ['bolero', 'boloro'],
                    'Cardigan': ['cardigan'],
                    'Overall': ['avarol', 'overall', 'overoll', 'overoul','avaroul','ovaroul'],
                    'Shirt': ['shirt', 'shirts', 'chemise', 'amis', 'chirt', 'amis kim tawil', 'amis kim taweel'],
                    'Pantacour': ['pantacour', 'pantacoor', 'pantacor'],
                    'Skirt': ['skirt', 'skirts'],
                    'Abaya': ['abaya', 'gown'],
                    'Body piece': ['body', 'bady'],
                    'Corole': ['corole','corolle','colrolet'],
                    'Pavette': ['bavette', 'pavette', 'bavte', 'pavete','bavat'],
                    'Towel': ['towel', 'manshafe', 'towels','toweal'],
                    'Bag': ['bag', 'bags', 'shanta'],
                    'Bed sheet': ['charchaf', 'charchf', 'charchef', 'sharshaf', 'bedsheet', 'bed sheet', 'sheet','charcef','charchaf','charcf','charef','charcaf','charchaf mejweiz','sheet set double'],
                    'Blanket': ['blanket', 'hrem', 'hram'],
                    'Pillow': ['mkade', 'tikayeh', 'tikkayeh', 'takiye', 'pillow','mkhade'],
                    'Pillow covers': ['sac mkada', 'pillow case', 'pillow sheet','sac mkhade'],
                    'Scarf': ['chall', 'shall', 'chal','shal'],
                    'Kitchen item': ['kitchen towel', 'matbakh','kitchen towels','kithcen maryoul'],
                    'Barbotese': ['barbotese','barboties','barbotise','cross','drsyer','berbotese'],
                    'Baby Nest': ['nest'],
                    'Baby Changing Mat': ['lataa','mate','mate set 2pcs'],
                    'Baby Covering Blanket': ['mlafe','mlafeh','slib','mlaffeh','mlaffe'],
                    'Maryoul': ['maryoul'],
                    'Tie': ['gravatte'],
                    'Joggings': ['jogg','jogging','joggings'],
                    'Ear Cover': ['ear'],
                    'Baby Napkin': ['napkin'],

                }   

            # Convert the product name to lowercase for case-insensitive comparison
            product_name_lower = product_name.lower()

            # Check for specific combined phrases first
            for category, keywords in categories.items():
                    if category in ['Cap + Scarf', 'Cap + Gloves','Kitchen item','Pyjamas','Swimwear','Bed sheet','Baby Changing Mat']:
                            for keyword in keywords:
                                if keyword in product_name_lower:
                                    return category

            # Check for individual keywords
            for category, keywords in categories.items():
                    if category not in ['Cap + Scarf', 'Cap + Gloves','Kitchen item','Set','Swimwear','Bed sheet','Baby Changing Mat']:
                            for keyword in keywords:
                                if keyword in product_name_lower:
                                    return category

                # If no match found
            return 'Others'
            
        # Apply the categorization function to create the 'Product Type' column
        product_sales_dollar['Product Type'] = product_sales_dollar['Product name'].apply(categorize_product)


        ## Generic Categorization 

        # Define a function to map 'Product Type' to 'Product Category'
        def map_product_category(product_type):
                product_type_to_category = {
                    'Cap': 'Headwear',
                    'Hat': 'Headwear',
                    'Hijab': 'Headwear',
                    'Scarf': 'Headwear',
                    'Ear Cover': 'Headwear',
                    'Cap + Scarf': 'Headwear',
                    'Cap + Gloves': 'Headwear',
                    'Blouse': 'Top',
                    'T-shirt': 'Top',
                    'Sweater': 'Top',
                    'Hoodie': 'Top',
                    'Jacket': 'Top',
                    'Gillet': 'Top',
                    'Bolero': 'Top',
                    'Cardigan': 'Top',
                    'Shirt': 'Top',
                    'Pants': 'Bottom',
                    'Jeans': 'Bottom',
                    'Trousers': 'Bottom',
                    'Leggings': 'Bottom',
                    'Skirt': 'Bottom',
                    'Pantacour': 'Bottom',
                    'Short': 'Bottom',
                    'Slippers': 'Footwear',
                    'Socks': 'Footwear',
                    'Shoes': 'Footwear',
                    'Collant': 'Footwear',
                    'Kitchen item': 'Kitchen item',
                    'Bag': 'Bags',
                    'Boxer': 'Underwear',
                    'Slip Underwear': 'Underwear',
                    'Barbotese': 'Underwear',
                    'Bra': 'Underwear',
                    'Vest': 'Underwear',
                    'Bed sheet': 'Bed items',
                    'Blanket': 'Bed items',
                    'Pillow': 'Pillows',
                    'Pillow covers': 'Pillows',
                    'Towel': 'Bathroom item',
                    'Gloves': 'Additional Clothing',
                    'Belt': 'Additional Clothing',
                    'Pavette': 'Additional Clothing',
                    'Tie': 'Additional Clothing',
                    'Others': 'Others',
                    'Overall': 'Full Body Clothing',
                    'Dress': 'Full Body Clothing',
                    'Maryoul': 'Full Body Clothing',
                    'Robe': 'Full Body Clothing',
                    'Pyjamas': 'Full Body Clothing',
                    'Corole': 'Full Body Clothing',
                    'Body piece': 'Full Body Clothing',
                    'Set': 'Full Body Clothing',
                    'Joggings': 'Full Body Clothing',
                    'Abaya': 'Full Body Clothing',
                    'Beach item': 'Beach Wear',
                    'Swimwear': 'Beach Wear',
                    'Baby Nest': 'Baby Accessories',
                    'Baby Changing Mat': 'Baby Accessories',
                    'Baby Covering Blanket': 'Baby Accessories',
                    'Baby Napkin': 'Baby Accessories'
                }
            
                return product_type_to_category.get(product_type)
            
        # Using  'map_product_category' function to create the 'Product Category' column based on 'Product Type'
        product_sales_dollar['Product Category'] = product_sales_dollar['Product Type'].apply(map_product_category) 


        ## Adding in-depth date columns

        # Extracting different date-time components
        product_sales_dollar['dayofweek'] = product_sales_dollar['Date'].dt.dayofweek   
        product_sales_dollar['quarter'] = product_sales_dollar['Date'].dt.quarter
        product_sales_dollar['month'] =product_sales_dollar['Date'].dt.month
        product_sales_dollar['year'] = product_sales_dollar['Date'].dt.year
        product_sales_dollar['dayofyear'] = product_sales_dollar['Date'].dt.dayofyear
        product_sales_dollar['dayofmonth'] = product_sales_dollar['Date'].dt.day
        # Assuming the week numbering starts from 1
        product_sales_dollar['weekofyear'] = product_sales_dollar['Date'].dt.isocalendar().week  

        ## Adding holiday columns
        # Convert the 'Date' column to datetime format
        product_sales_dollar['Date'] = pd.to_datetime(product_sales['Date'])

        # Define holiday dates for each year
        holidays_2021 = pd.to_datetime(['2021-05-13', '2021-07-20', '2021-07-21', '2021-12-25', '2021-01-06', '2021-04-04', '2021-05-02'])
        holidays_2022 = pd.to_datetime(['2022-05-03', '2022-07-10', '2022-07-11', '2022-12-25', '2022-01-06', '2022-04-17', '2022-04-18', '2022-04-24', '2022-04-25'])
        holidays_2023 = pd.to_datetime(['2023-04-21', '2023-04-22', '2023-06-28', '2023-06-29', '2023-12-25', '2023-01-06', '2023-04-09', '2023-04-16'])

        # Combine all holiday dates into a single list
        all_holidays = holidays_2021.to_list() + holidays_2022.to_list() + holidays_2023.to_list()

        # Convert the list to a set for faster membership checking
        all_holidays_set = set(all_holidays)

        # Function to determine if a date is a holiday or not
        def is_holiday(date):
            return "holiday" if date in all_holidays_set else "normal day"

        # Apply the function to create the "holidays and seasons" column
        product_sales_dollar['Holidays'] = product_sales['Date'].apply(is_holiday)


        ## Naming Holidays and Seasonalities

        # Define the date ranges for each holiday and season
        holidays_and_seasons = [
            ("Back to school season", datetime(2023, 8, 25), datetime(2023, 9, 25)),
            ("New year and christmas season", datetime(2023, 12, 18), datetime(2023, 12, 31)),
            ("Ramadan Day", datetime(2021, 4, 12), datetime(2021, 5, 12)),
                ("Ramadan Day", datetime(2022, 4, 2), datetime(2022, 5, 1)),
                ("Ramadan Day", datetime(2023, 3, 22), datetime(2023, 4, 20)),
                ("Christmas ", datetime(2021, 12, 25), datetime(2021, 12, 25)),
                ("Christmas ", datetime(2022, 12, 25), datetime(2022, 12, 25)),
                ("Christmas", datetime(2023, 12, 25), datetime(2023, 12, 25)),
                ("Armenian Orthodox Christmas", datetime(2021, 1, 6), datetime(2021, 1, 6)),
                ("Armenian Orthodox Christmas", datetime(2022, 1, 6), datetime(2022, 1, 6)),
                ("Armenian Orthodox Christmas", datetime(2023, 1, 6), datetime(2023, 1, 6)),
                ("Eid el Adha ", datetime(2021, 7, 20), datetime(2021, 7, 21)),
                ("Eid el Adha ", datetime(2022, 7, 10), datetime(2022, 7, 11)),
                ("Eid el Adha ", datetime(2023, 6, 28), datetime(2023, 6, 29)),
                ("Eid el Fetr", datetime(2021, 5, 13), datetime(2021, 5, 13)),
                ("Eid el Fetr ", datetime(2022, 5, 3), datetime(2022, 5, 3)),
                ("Eid el Fetr ", datetime(2023, 4, 21), datetime(2023, 4, 22)),
                ("Easter Sunday", datetime(2021, 4, 4), datetime(2021, 4, 4)),
                ("Easter Sunday", datetime(2022, 4, 17), datetime(2022, 4, 17)),
                ("Easter Sunday", datetime(2023, 4, 9), datetime(2023, 4, 9)),
                ("Orthodox Easter", datetime(2021, 5, 2), datetime(2021, 5, 2)),
                ("Orthodox Easter ", datetime(2022, 4, 24), datetime(2022, 4, 24)),
                ("Orthodox Easter ", datetime(2023, 4, 16), datetime(2023, 4, 16)),
        ]

        # Convert the date ranges into pandas Timestamps
        holidays_and_seasons = [(name, pd.Timestamp(start), pd.Timestamp(end)) for name, start, end in holidays_and_seasons]

        # Function to check if the date falls within any date range
        def get_holiday_season_name(date):
                for name, start, end in holidays_and_seasons:
                    if start <= date <= end:
                        return name
                return "Regular day"  # If no holiday or season is found, return "Regular day"

        # Add a new column to the data frame
        product_sales_dollar["Holidays and Seasons Names"] = product_sales_dollar["Date"].apply(get_holiday_season_name)


        ## Adding Month Names 
        # Add a 'Month' column with full month names
        product_sales_dollar['Month'] = product_sales_dollar['Date'].dt.strftime('%B')

        #adding week day names
        # Add a 'Day_of_Week' column
        product_sales_dollar['Day_of_Week'] = product_sales_dollar['Date'].dt.day_name()

        processed_data = product_sales_dollar

        return processed_data
########################################################################################################################################################################
########################################################################################################################################################################


########################################################################################################################################################################
########################################################################################################################################################################
# Function to prepare and run Market Basket Analysis
def run_market_basket_analysis(df2,min_support):
    df2 = df2[['Invoice','Product name',  'Quantity', 'Age Group', 'Gender']]
     # Define a function to categorize the product names
    def categorize_product(row):
            # Mapping of categories to respective keywords (all in lowercase)
            categories = {
                'Belt': ['belt', 'belts', 'zennar','zenar','zinnar'],
                'Blouse': ['blouse', 'bloue', 'bloues','blousr'],
                'T-shirt': ['t-shirt', 'tshirt', 'tchirt', 'tchrt'],
                'Sweater': ['sweater', 'sweeter', 'sweatcher', 'pullover','kim tawil','kim taweel'],
                'Hoodie': ['hoodie', 'hoodies', 'hoody'],
                'Boxer': ['boxer', 'boxers', 'bxr', 'brief','breif'],
                'Slip Underwear': ['underwear', 'kilot', 'string', 'sleep', 'slip', 'slips', 'sleeps', 'under', 'ladies night dr', 'twinz', 'twins'],
                'Bra': ['bra', 'bras', 'soutian', 'soutien','sotian', 'brassires','soyien','sotien','soutin','soutien', 'soutein'],
                'Jacket': ['jacket', 'jaket', 'jacet', 'jac'],
                'Short': ['short', 'shorts','shourt','chort','chourt'],
                'Dress': ['dress', 'drss'],
                'Gillet': ['gilett', 'gillet', 'jillet', 'jilet', 'gilet', 'jilit','jiliet','jillit',  'jiliiet'],
                'Hijab': ['isharb', 'thjeb'],
                'Pants': ['pnt', 'pant', 'pantalon', 'pantalon', 'pants'],
                'Robe': ['bornos', 'bournos', 'bornous', 'bournous', 'robe'],
                'Jeans': ['jens', 'jean', 'jeans', 'jins', 'jns', 'jeanes','jaens'],
                'Trousers': ['trouser', 'trousers', 'trousier'],
                'Vest': ['vest', 'vests', 'bretel', 'brettel', 'brutel', 'brotel', 'broutel'],
                'Pyjamas': ['pyjama', 'pyjamas', 'pygama', 'pygamas', 'pyjam', 'pygam', 'pygamy', 'pygami', 'pjm','pyagama','pyajama'],
                'Shoes': ['shoe', 'shoes'],
                'Slippers': ['slipper', 'slippers', 'sliper', 'slipers', 'sleeper'],
                'Socks': ['sock', 'socks', 'sox', 'socs', 'soc', 'sok', 'soks'],
                'Collant': ['collant', 'collants', 'colant', 'colants', 'collont', 'colont', 'collonts', 'colonts','collon','collan'],
                'Leggings': ['legging', 'leggings', 'fizo', 'fizou', 'fuseau', 'vizou'],
                'Set': ['set', 'sets',],
                'Beach item': ['beach'],
                'Swimwear': ['mayo', 'swim', 'swimming','mayyo'],
                'Cap': ['cap', 'kab', 'kap'],
                'Hat': ['hat'],
                'Gloves': ['kaf', 'glove', 'gloves'],
                'Cap + Scarf': ['cap with shall', 'cap with chall', 'cap with shal', 'cap with chal',' cap + shall', 'cap + chall'],
                'Cap + Gloves': ['cap + glove', 'cap with kaf','cap with gloves', 'cap + kaf'],
                'Bolero': ['bolero', 'boloro'],
                'Cardigan': ['cardigan'],
                'Overall': ['avarol', 'overall', 'overoll', 'overoul','avaroul','ovaroul', 'overall15'],
                'Shirt': ['shirt', 'shirts', 'chemise', 'amis', 'chirt', 'amis kim tawil', 'amis kim taweel'],
                'Pantacour': ['pantacour', 'pantacoor', 'pantacor'],
                'Skirt': ['skirt', 'skirts'],
                'Abaya': ['abaya', 'gown'],
                'Body piece': ['body', 'bady'],
                'Corole': ['corole','corolle','colrolet'],
                'Pavette': ['bavette', 'pavette', 'bavte', 'pavete','bavat'],
                'Towel': ['towel', 'manshafe', 'towels','toweal'],
                'Bag': ['bag', 'bags', 'shanta'],
                'Bed sheet': ['charchaf', 'charchf', 'charchef', 'sharshaf', 'bedsheet', 'bed sheet', 'sheet'],
                'Blanket': ['blanket', 'hrem', 'hram'],
                'Pillow': ['mkade', 'tikayeh', 'tikkayeh', 'takiye', 'pillow','mkhade'],
                'Pillow covers': ['sac mkada', 'pillow case', 'pillow sheet','sac mkhade'],
                'Scarf': ['chall', 'shall', 'chal','shal'],
                'Kitchen item': ['kitchen towel', 'matbakh','kitchen towels','kithcen maryoul',  'charef477000', 'KITTCHEN3', 'kittchen'],
                'Barbotese': ['barbotese','barboties','barbotise','cross','drsyer','berbotese'],
                'Bedroom items': ['nest', 'charcef', 'PROTECTOR', 'charcf200','TAKKEYEH', 'charcf', 'charcaf', 'takkeyeh'],
                'Changing Mat': ['lataa','mate'],
                'Covering Blanket': ['mlafe','mlafeh','slib','mlaffeh','mlaffe', 'MLAFFEH'],
                'Maryoul': ['maryoul'],
                'Tie': ['gravatte'],
                'Joggings': ['jogg','jogging','joggings'],
                'Ear Cover': ['ear'],
                'Napkin': ['napkin', 'NAPKIN'],

            }


            # Extract values from 'Gender' and 'Age Group' columns
            gender = row['Gender']
            age_group = row['Age Group']

            # Convert the product name to lowercase for case-insensitive comparison
            product_name_lower = row['Product name'].lower()

            # Check for specific combined phrases first
            for category, keywords in categories.items():
                if category in ['Cap + Scarf', 'Cap + Gloves', 'Kitchen item', 'Covering Blanket', 'Slip Underwear']:
                    for keyword in keywords:
                        if keyword in product_name_lower:
                            return f"{category} - {gender} - {age_group}"

            # Check for individual keywords
            for category, keywords in categories.items():
                if category not in ['Cap + Scarf', 'Cap + Gloves', 'Kitchen item', 'Covering Blanket', 'Slip Underwear']:
                    for keyword in keywords:
                        if keyword in product_name_lower:
                            return f"{category} - {gender} - {age_group}"

            # If no match found
            return f"Others - {gender} - {age_group}"

    # Apply the categorization function to create the 'Description' column
    df2['Description'] = df2.apply(categorize_product, axis=1)

    others=df2[df2['Description'].str.startswith('Others', na=False)]

    

    df2 = df2[df2['Quantity'] >= 0]

    #Removing spaces from beginning and end
    df2['Description']= df2['Description'].str.strip()
    #Removing Missing Invoices
    df2.dropna(axis=0, subset=['Invoice'], inplace=True)
    #Converting invoice number to be a string
    df2['Invoice'] = df2['Invoice'].astype('str')
    #Removing the 'Others' Categories
    #myretaildata = myretaildata[~myretaildata['Description'].str.startswith('Others', na=False)]
    df2 = df2[df2['Description'].str.startswith('Others', na=False) == False]

    #Creating the basket
    mybasket = (df2.groupby(['Invoice', 'Description'])['Quantity']
                        .sum().unstack().reset_index().fillna(0)
                        .set_index('Invoice'))
    
    #Converting all positive values to 1 and everything else to 0
    def my_encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    my_basket_sets = mybasket.applymap(my_encode_units)

    #Generatig frequent itemsets
    my_frequent_itemsets = apriori(my_basket_sets, min_support=min_support, use_colnames=True)

    #Generating rules
    my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=1)

       # Viewing top rules
    return my_rules   

########################################################################################################################################################################
########################################################################################################################################################################
#Menu Bar 

# Creating Navigation Bar for streamlit app
#make it look nice from the start
st.set_page_config(layout='wide' ,page_title= 'Mahfouz Stores Sales Tool',
page_icon= '🛍️', initial_sidebar_state= 'expanded')

def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <h2 style = "color:#010101; text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "color:#010101; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)


# specify the primary menu definition
menu_data = [
    {'icon': "❔", 'label':"About",'ttip':"eda"},
    {'icon': "🔄", 'label':"Data Processing Tool",'ttip':"discover"},
    {'icon': "📈", 'label':"Forecasting Tool"},
    {'icon': "🧺", 'label':"Market Basket Analysis Tool"}
]

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


# Add your image above the menu bar
image_path = "Unknown.png"  # Replace with the actual path to your image
left,middle,right = st.columns((1,1,1))

with middle:
    st.image(image_path, use_column_width=True)

over_theme = {'menu_background':'#254C8E'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    sticky_nav=False, #at the top or not
    hide_streamlit_markers=False,
    sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
)


def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie= load_lottieurl('https://lottie.host/9d625c5c-be6b-46a4-9f0d-dc6498471153/nfUFwPHkgc.json')


########################################################################################################################################################################
########################################################################################################################################################################


if menu_id == 'About':
    st.markdown("<h1 style='text-align: center;'>Mahfouz Stores Sales Analysis Tool</h1>", unsafe_allow_html=True)

    left, right= st.columns((1,1))

    with left:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        display_app_header(main_txt='About the tool',
                   sub_txt="The following tool is a user-friendly analysis tool that allows you to process your store's data for analysis, while also utilizing Machine Learning to provide forecasting and Market Basket Analaysis techniques. It will aid Mahfouz stores in managing inventory, store logistics, and making future plans based on data-driven predictions and decisions. Additionally, the tool supports Bundle Creation and item in-shop positioning via Market baske Analaysis"
                   )
        
        st.write('')
        st.write('')
        display_app_header(main_txt='How to use the Tool',
                   sub_txt=" ● Data Processing Tool: After extracting the data, The data processing tool is accessed through the web app, requiring users to upload their data for cleaning and processing. "
                   
                   )

        st.markdown(" ● Forecasting Tool: After processing the data, upload the processed data to the forecasting tool, and then select the desired forecast type. " ) 

        st.markdown(" ● Market Basket Analaysis Tool: Access the tool through the web app navigator bar above, run the analysis and the provided combinations with their assoication values will appear" )
                   

        

        
        


    with right:
        st_lottie(lottie, height= 600,key= 'coding')


########################################################################################################################################################################

# Processing tool page
if menu_id == 'Data Processing Tool':



    # Function to generate a download link for CSV files
    def get_csv_download_link(csv_file):
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download processed data</a>'
        return href
    st.markdown("<h1 style='text-align: center;'>Data Processing Tool</h1>", unsafe_allow_html=True)

    # File upload section for product sales data
    product_sales_data = st.file_uploader("Upload Product Sales CSV file", type=['csv'])


    if product_sales_data is not None:
    #Check if the user has triggered processing
        if st.button('Process Data'):
            # Call the data processing function with the uploaded product sales file and dollar rate data
            processed_data = process_data(product_sales_data)

            # Store processed_data in session state
            st.session_state.processed_data = processed_data

            # Display the processed data
            st.subheader('Processed Data:')
            st.write(processed_data)


            # Download link for processed data
            csv_file = processed_data.to_csv(index=False)
            st.markdown(get_csv_download_link(csv_file), unsafe_allow_html=True)


########################################################################################################################################################################

df= pd.read_csv("Mahfouz Fixed Data1.csv")
if menu_id == 'Forecasting Tool':
    # Convert the 'Date' column to datetime type
    # Change the format of the 'Date' column to month-day-year ("%m-%d-%Y")
    df['Date'] = df['Date'].str.split('/').apply(lambda x: f"{x[1]}-{x[0]}-{x[2]}")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    st.markdown("<h1 style='text-align: center;'>Demand Forecast</h1>", unsafe_allow_html=True)

    # Display Forecast Options based on user selection
    if st.button("Total Quantity Forecast"):
        st.subheader('Total Quantity Forecast')

        # Drop rows where Quantity is less than 0
        df = df[df['Quantity'] >= 0]

        # Selecting specific columns
        df = df[['Date','Quantity']]

        # Grouping data by date and summing the 'quantity' column
        df = df.groupby('Date')['Quantity'].sum().reset_index() 

        # Convert 'DATE' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Set min_date to January 1, 2021
        min_date = pd.Timestamp('2021-01-01')
        max_date = df['Date'].max()
        date_range = pd.date_range(start= min_date, end=max_date, freq='D')

        # Create a DataFrame with the complete date range
        complete_df = pd.DataFrame({'Date': date_range})

        # Merge the complete date range DataFrame with your original DataFrame to fill in missing dates
        merged_df = pd.merge(complete_df, df, on='Date', how='left')

        # Calculate average quantity for each month and round the values
        merged_df['month'] = merged_df['Date'].dt.to_period('M')
        monthly_avg = merged_df.groupby('month')['Quantity'].mean().round()

        # Fill missing quantity values with rounded respective month's average
        merged_df['Quantity'] = merged_df['Quantity'].fillna(merged_df['month'].map(monthly_avg))

        # Drop the 'month' column created for calculation purposes
        merged_df.drop('month', axis=1, inplace=True)

        # Displaying the updated DataFrame
        df = merged_df

        # Convert 'Date' column to datetime if it's not already in datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Extract various date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day_of_week'] = df['Date'].dt.dayofweek  # Monday: 0, Sunday: 6
        df['Day_of_month'] = df['Date'].dt.day
        df['Quarter'] = df['Date'].dt.quarter

        # Convert the 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Define holiday dates for each year
        holidays_2021 = pd.to_datetime(['2021-05-13', '2021-07-20', '2021-07-21', '2021-12-25', '2021-01-06', '2021-04-04', '2021-05-02','2021-08-15','2021-12-25','2021-11-22','2021-08-04'])
        holidays_2022 = pd.to_datetime(['2022-05-03', '2022-07-10', '2022-07-11', '2022-12-25', '2022-01-06', '2022-04-17', '2022-04-18', '2022-04-24', '2022-04-25','2022-08-15','2022-08-04'])
        holidays_2023 = pd.to_datetime(['2023-04-21', '2023-04-22', '2023-06-28', '2023-06-29', '2023-12-25', '2023-01-06', '2023-04-09', '2023-04-16','2023-11-22','2023-12-31','2023-07-19','2023-06-29','2023-06-28','2023-07-28','2023-08-04','2023-11-22','2023-08-15'])

        # Combine all holiday dates into a single list
        all_holidays = holidays_2021.to_list() + holidays_2022.to_list() + holidays_2023.to_list()

        # Convert the list to a set for faster membership checking
        all_holidays_set = set(all_holidays)

        # Function to determine if a date is a holiday or not
        def is_holiday(date):
            return "1" if date in all_holidays_set else "0"

        # Apply the function to create the "holidays and seasons" column
        df['Holidays'] = df['Date'].apply(is_holiday)

    
        # Convert 'Date' column to datetime if it's not already in datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Extract the 'Month' from the 'Date'
        df['Month'] = df['Date'].dt.month

        # Define a function to map months to seasons
        def get_season(month):
            if month in [9, 10, 11]:
                return 'Fall'
            elif month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            else:  # Months 6, 7, 8
                return 'Summer'

        # Apply the function to create the 'Season' column
        df['Season'] = df['Month'].apply(get_season)

        from scipy.stats import zscore


        # Calculate z-scores for 'Quantity' column
        df['Quantity_ZScore'] = zscore(df['Quantity'])

        # Define a threshold for outliers (e.g., Z-score greater than 3 or -3)
        outlier_threshold = 3

        # Replace outliers in 'Quantity' column with mean value
        mean_quantity = df['Quantity'].mean()
        df.loc[df['Quantity_ZScore'].abs() > outlier_threshold, 'Quantity'] = mean_quantity

        # Drop the temporary Z-score column if no longer needed
        df.drop('Quantity_ZScore', axis=1, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'])

        # Separate features and target variable
        X = df[['Year', 'Month', 'Day_of_week', 'Day_of_month','Quarter', 'Season','Holidays']]  # Features + quarter then season then + 'holidays
        y = df['Quantity']  # Target variable

        # Convert categorical columns to dummy variables
        X = pd.get_dummies(X, columns=['Season', 'Holidays'])

        # Define the date ranges for training and testing
        train_start_date = pd.to_datetime('2021-01-04')
        train_end_date = pd.to_datetime('2023-06-04')
        test_start_date = pd.to_datetime('2023-06-05')

        # Split the data into train and test sets based on the date ranges
        train_mask = (df['Date'] >= train_start_date) & (df['Date'] <= train_end_date)
        test_mask = df['Date'] >= test_start_date

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Initialize and train the XGBoost model
        model = XGBRegressor(learning_rate=0.03, max_depth=3, n_estimators=155)
        from sklearn.model_selection import cross_val_score
        from xgboost import XGBRegressor

        model.fit(X_train, y_train)

        # Generate dates from December 10th to December 31st, 2023
        dates_2023 = pd.date_range(start='2023-12-15', end='2023-12-31', freq='D')

        # Create a DataFrame with only the 'Date' column
        df23 = pd.DataFrame({'Date': dates_2023})

        # Convert 'Date' column to datetime if it's not already in datetime format
        df23['Date'] = pd.to_datetime(df23['Date'])

        # Extract various date features
        df23['Year'] = df23['Date'].dt.year
        df23['Month'] = df23['Date'].dt.month
        df23['Day_of_week'] = df23['Date'].dt.dayofweek  # Monday: 0, Sunday: 6
        df23['Day_of_month'] = df23['Date'].dt.day
        df23['Quarter'] = df23['Date'].dt.quarter

        df23['Date'] = pd.to_datetime(df23['Date'])

        # Define holiday dates for each year
        holidays_2021 = pd.to_datetime(['2021-05-13', '2021-07-20', '2021-07-21', '2021-12-25', '2021-01-06', '2021-04-04', '2021-05-02','2021-08-15','2021-12-25','2021-11-22','2021-08-04'])
        holidays_2022 = pd.to_datetime(['2022-05-03', '2022-07-10', '2022-07-11', '2022-12-25', '2022-01-06', '2022-04-17', '2022-04-18', '2022-04-24', '2022-04-25','2022-08-15','2022-08-04'])
        holidays_2023 = pd.to_datetime(['2023-04-21', '2023-04-22', '2023-06-28', '2023-06-29', '2023-12-25', '2023-01-06', '2023-04-09', '2023-04-16','2023-11-22','2023-12-31','2023-07-19','2023-06-29','2023-06-28','2023-07-28','2023-08-04','2023-11-22','2023-08-15'])

        # Combine all holiday dates into a single list
        all_holidays = holidays_2021.to_list() + holidays_2022.to_list() + holidays_2023.to_list()

        # Convert the list to a set for faster membership checking
        all_holidays_set = set(all_holidays)

        # Function to determine if a date is a holiday or not
        def is_holiday(date):
            return "1" if date in all_holidays_set else "0"

        # Apply the function to create the "holidays and seasons" column
        df23['Holidays'] = df23['Date'].apply(is_holiday)

        # Convert 'Date' column to datetime if it's not already in datetime format
        df23['Date'] = pd.to_datetime(df23['Date'])

        # Extract the 'Month' from the 'Date'
        df23['Month'] = df23['Date'].dt.month

        # Define a function to map months to seasons
        def get_season(month):
            if month in [9, 10, 11]:
                return 'Fall'
            elif month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            else:  # Months 6, 7, 8
                return 'Summer'

        # Apply the function to create the 'Season' column
        df23['Season'] = df23['Month'].apply(get_season)

        # Add a 'Quantity' column filled with zeros as a placeholder
        df23['Quantity'] = 0

        # Separate features and target variable
        X23 = df23[['Year', 'Month', 'Day_of_week', 'Day_of_month','Quarter', 'Season','Holidays']]  # Features + quarter then season then  + holidays

        # Convert categorical columns to dummy variables
        X23 = pd.get_dummies(X23, columns=['Season','Holidays']) #+ holidays

        X23['Season_Spring'] = 0
        X23['Season_Summer'] = 0
        X23['Season_Fall'] = 0

        X23 = pd.DataFrame({

            'Year': X23['Year'],
            'Month': X23['Month'],
            'Day_of_week': X23['Day_of_week'],
            'Day_of_month': X23['Day_of_month'],
            'Quarter': X23['Quarter'], 
            'Season_Fall': X23['Season_Fall'],
            'Season_Spring': X23['Season_Spring'],
            'Season_Summer': X23['Season_Summer'],
            'Season_Winter': X23['Season_Winter'],
            'Holidays_0': X23['Holidays_0'],
            'Holidays_1': X23['Holidays_1'],

        })

        # Make predictions using the trained XGBoost model
        y_pred23 = model.predict(X23)

        # Assign the predictions to the 'Quantity' column in the new_dates_2023 DataFrame
        df23['Quantity'] = y_pred23

        # Convert specific columns of forecast_df to NumPy arrays
        date_array = df23['Date'].values
        forecast_array = df23['Quantity'].values

        # Plotting forecast vs. actual using Streamlit and Matplotlib
        fig, ax = plt.subplots(figsize=(15, 3))
        ax.plot(date_array, forecast_array, label='Forecast', linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Quantity')
        ax.set_title('XGBoost Forecast vs. Actual')
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)

    else:

        st.subheader('Category-wise Forecast')

        selected_category = st.selectbox('Select Category', df['Product Category'].unique())

        if st.button("Category Forecast level"):

            # Convert 'Date' column to datetime if it's not already in datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Filter rows for Product Category 
            cat_data = df[df['Product Category'] == selected_category]

            # Group by 'Date' and sum the 'Quantity' to get total underwear purchased on each day
            cat_daily_quantity = cat_data.groupby('Date')['Quantity'].sum().reset_index()

            # Create a DataFrame 'df_underwear' with Date and Quantity columns
            df_cat = pd.DataFrame(cat_daily_quantity, columns=['Date', 'Quantity'])

            # Fill in missing dates with a quantity of 0 to match the original date span
            date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
            missing_dates = set(date_range) - set(df_cat['Date'])
            missing_dates_df = pd.DataFrame({'Date': list(missing_dates), 'Quantity': 0})

            # Concatenate missing dates with quantity 0 to df_underwear
            df = pd.concat([df_cat, missing_dates_df]).sort_values('Date').reset_index(drop=True)
            df.columns = ['Date', 'Quantity']

            # Convert 'Date' column to datetime if it's not already in datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Extract various date features
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day_of_week'] = df['Date'].dt.dayofweek  # Monday: 0, Sunday: 6
            df['Day_of_month'] = df['Date'].dt.day
            df['Quarter'] = df['Date'].dt.quarter

            # Convert the 'Date' column to datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Define holiday dates for each year
            holidays_2021 = pd.to_datetime(['2021-05-13', '2021-07-20', '2021-07-21', '2021-12-25', '2021-01-06', '2021-04-04', '2021-05-02','2021-08-15','2021-12-25','2021-11-22','2021-08-04'])
            holidays_2022 = pd.to_datetime(['2022-05-03', '2022-07-10', '2022-07-11', '2022-12-25', '2022-01-06', '2022-04-17', '2022-04-18', '2022-04-24', '2022-04-25','2022-08-15','2022-08-04'])
            holidays_2023 = pd.to_datetime(['2023-04-21', '2023-04-22', '2023-06-28', '2023-06-29', '2023-12-25', '2023-01-06', '2023-04-09', '2023-04-16','2023-11-22','2023-12-31','2023-07-19','2023-06-29','2023-06-28','2023-07-28','2023-08-04','2023-11-22','2023-08-15'])

            # Combine all holiday dates into a single list
            all_holidays = holidays_2021.to_list() + holidays_2022.to_list() + holidays_2023.to_list()

            # Convert the list to a set for faster membership checking
            all_holidays_set = set(all_holidays)

            # Function to determine if a date is a holiday or not
            def is_holiday(date):
                return "1" if date in all_holidays_set else "0"

            # Apply the function to create the "holidays and seasons" column
            df['Holidays'] = df['Date'].apply(is_holiday)

        
            # Convert 'Date' column to datetime if it's not already in datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Extract the 'Month' from the 'Date'
            df['Month'] = df['Date'].dt.month

            # Define a function to map months to seasons
            def get_season(month):
                if month in [9, 10, 11]:
                    return 'Fall'
                elif month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                else:  # Months 6, 7, 8
                    return 'Summer'

            # Apply the function to create the 'Season' column
            df['Season'] = df['Month'].apply(get_season)

            from scipy.stats import zscore


            # Calculate z-scores for 'Quantity' column
            df['Quantity_ZScore'] = zscore(df['Quantity'])

            # Define a threshold for outliers (e.g., Z-score greater than 3 or -3)
            outlier_threshold = 3

            # Replace outliers in 'Quantity' column with mean value
            mean_quantity = df['Quantity'].mean()
            df.loc[df['Quantity_ZScore'].abs() > outlier_threshold, 'Quantity'] = mean_quantity

            # Drop the temporary Z-score column if no longer needed
            df.drop('Quantity_ZScore', axis=1, inplace=True)

            df['Date'] = pd.to_datetime(df['Date'])

            # Separate features and target variable
            X = df[['Year', 'Month', 'Day_of_week', 'Day_of_month','Quarter', 'Season','Holidays']]  # Features + quarter then season then + 'holidays
            y = df['Quantity']  # Target variable

            # Convert categorical columns to dummy variables
            X = pd.get_dummies(X, columns=['Season', 'Holidays'])

            # Define the date ranges for training and testing
            train_start_date = pd.to_datetime('2021-01-04')
            train_end_date = pd.to_datetime('2023-06-04')
            test_start_date = pd.to_datetime('2023-06-05')

            # Split the data into train and test sets based on the date ranges
            train_mask = (df['Date'] >= train_start_date) & (df['Date'] <= train_end_date)
            test_mask = df['Date'] >= test_start_date

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # Initialize and train the XGBoost model
            model1 = XGBRegressor(learning_rate=0.035, max_depth=3, n_estimators=175)
            from sklearn.model_selection import cross_val_score
            from xgboost import XGBRegressor

            model1.fit(X_train, y_train)

            # Generate dates from December 10th to December 31st, 2023
            dates_2023 = pd.date_range(start='2023-12-15', end='2023-12-31', freq='D')

            # Create a DataFrame with only the 'Date' column
            df23 = pd.DataFrame({'Date': dates_2023})

            # Convert 'Date' column to datetime if it's not already in datetime format
            df23['Date'] = pd.to_datetime(df23['Date'])

            # Extract various date features
            df23['Year'] = df23['Date'].dt.year
            df23['Month'] = df23['Date'].dt.month
            df23['Day_of_week'] = df23['Date'].dt.dayofweek  # Monday: 0, Sunday: 6
            df23['Day_of_month'] = df23['Date'].dt.day
            df23['Quarter'] = df23['Date'].dt.quarter

            df23['Date'] = pd.to_datetime(df23['Date'])

            # Define holiday dates for each year
            holidays_2021 = pd.to_datetime(['2021-05-13', '2021-07-20', '2021-07-21', '2021-12-25', '2021-01-06', '2021-04-04', '2021-05-02','2021-08-15','2021-12-25','2021-11-22','2021-08-04'])
            holidays_2022 = pd.to_datetime(['2022-05-03', '2022-07-10', '2022-07-11', '2022-12-25', '2022-01-06', '2022-04-17', '2022-04-18', '2022-04-24', '2022-04-25','2022-08-15','2022-08-04'])
            holidays_2023 = pd.to_datetime(['2023-04-21', '2023-04-22', '2023-06-28', '2023-06-29', '2023-12-25', '2023-01-06', '2023-04-09', '2023-04-16','2023-11-22','2023-12-31','2023-07-19','2023-06-29','2023-06-28','2023-07-28','2023-08-04','2023-11-22','2023-08-15'])

            # Combine all holiday dates into a single list
            all_holidays = holidays_2021.to_list() + holidays_2022.to_list() + holidays_2023.to_list()

            # Convert the list to a set for faster membership checking
            all_holidays_set = set(all_holidays)

            # Function to determine if a date is a holiday or not
            def is_holiday(date):
                return "1" if date in all_holidays_set else "0"

            # Apply the function to create the "holidays and seasons" column
            df23['Holidays'] = df23['Date'].apply(is_holiday)

            # Convert 'Date' column to datetime if it's not already in datetime format
            df23['Date'] = pd.to_datetime(df23['Date'])

            # Extract the 'Month' from the 'Date'
            df23['Month'] = df23['Date'].dt.month

            # Define a function to map months to seasons
            def get_season(month):
                if month in [9, 10, 11]:
                    return 'Fall'
                elif month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                else:  # Months 6, 7, 8
                    return 'Summer'

            # Apply the function to create the 'Season' column
            df23['Season'] = df23['Month'].apply(get_season)

            # Add a 'Quantity' column filled with zeros as a placeholder
            df23['Quantity'] = 0

            # Separate features and target variable
            X23 = df23[['Year', 'Month', 'Day_of_week', 'Day_of_month','Quarter', 'Season','Holidays']]  # Features + quarter then season then  + holidays

            # Convert categorical columns to dummy variables
            X23 = pd.get_dummies(X23, columns=['Season','Holidays']) #+ holidays

            X23['Season_Spring'] = 0
            X23['Season_Summer'] = 0
            X23['Season_Fall'] = 0

            X23 = pd.DataFrame({

                'Year': X23['Year'],
                'Month': X23['Month'],
                'Day_of_week': X23['Day_of_week'],
                'Day_of_month': X23['Day_of_month'],
                'Quarter': X23['Quarter'], 
                'Season_Fall': X23['Season_Fall'],
                'Season_Spring': X23['Season_Spring'],
                'Season_Summer': X23['Season_Summer'],
                'Season_Winter': X23['Season_Winter'],
                'Holidays_0': X23['Holidays_0'],
                'Holidays_1': X23['Holidays_1'],

            })

            # Make predictions using the trained XGBoost model
            y_pred23 = model1.predict(X23)

            # Assign the predictions to the 'Quantity' column in the new_dates_2023 DataFrame
            df23['Quantity'] = y_pred23

            # Convert specific columns of forecast_df to NumPy arrays
            date_array = df23['Date'].values
            forecast_array = df23['Quantity'].values

            # Plotting forecast vs. actual using Streamlit and Matplotlib
            fig, ax = plt.subplots(figsize=(15, 3))
            ax.plot(date_array, forecast_array, label='Forecast', linestyle='--')
            ax.set_xlabel('Date')
            ax.set_ylabel('Quantity')
            ax.set_title('XGBoost Forecast vs. Actual')
            ax.legend()

            # Display the plot in Streamlit
            st.pyplot(fig)
        else:
            st.warning("No Forecast level chosen")   


######################################################################################################################################################################## 
if menu_id == 'Market Basket Analysis Tool':
    st.markdown("<h1 style='text-align: center;'>Market Basket Analysis Tool</h1>", unsafe_allow_html=True)
     
         # Check if processed data exists in session state
    if 'processed_data' in st.session_state:
        # Display the processed data
        # Assign processed data to df2
        df2 = st.session_state.processed_data

    # Input fields for setting min_support and number of rows
    # Generate a sequence of min_support values
    min_support_values = [0.01] + [round(0.01 + i * 0.005, 3) for i in range(1, 8)]

    # Get user input for selecting min_support value
    # Generate a sequence of min_support values
    min_support_values = [0.01] + [round(0.01 + i * 0.005, 3) for i in range(1, 8)]

    # Create a dropdown for min_support selection
    min_support = st.number_input('Enter min_support value', min_value=0.010, max_value=0.05, step=0.005, format="%.3f", value=min_support_values[0], key="min_support")

    if st.button("Run Market Basket Analysis"):
        if 'processed_data' in st.session_state:
            df2 = st.session_state.processed_data

            # Run Market Basket Analysis function with user-defined min_support
            rules = run_market_basket_analysis(df2, min_support)

            # Display the results
            st.write("Top Rules:")
            st.write(rules)
        else:
            st.warning("No processed data available. Please process data in the 'Data Processing Tool' page.")




