import pandas as pd

# Read in our dataset
# It's sensitive to spaces in the CSV, so no spaces allowed
input_file = "adult.data"
df = pd.read_csv(input_file, header=0)

work_class_map = {"Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3, "Local-gov": 4,
                  "State-gov": 5, "Without-pay": 6, "Never-worked": 7}

marital_status_map = {"Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2, "Separated": 3, "Widowed": 4,
                      "Married-spouse-absent": 5, "Married-AF-spouse": 6}

occupation_map = {"?": -1, "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4,
                  "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8,
                  "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12,
                  "Armed-Forces": 13}

relationship_map = {"Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5}

race_map = {"White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2, "Other": 3, "Black": 4}

sex_map = {"Female": 0, "Male": 1}

native_country_map = {"?": -1, "United-States": 0, "Cambodia": 1, "England": 2, "Puerto-Rico": 3, "Canada": 4, "Germany": 5,
                  "Outlying-US(Guam-USVI-etc)": 6, "India": 7, "Japan": 8, "Greece": 9, "South": 10, "China": 11,
                  "Cuba": 12, "Iran": 13, "Honduras": 14, "Philippines": 15, "Italy": 16, "Poland": 17, "Jamaica": 18,
                  "Vietnam": 19, "Mexico": 20, "Portugal": 21, "Ireland": 22, "France": 23, "Dominican-Republic": 24,
                  "Laos": 25, "Ecuador": 26, "Taiwan": 27, "Haiti": 28, "Columbia": 29, "Hungary": 30, "Guatemala": 31,
                  "Nicaragua": 32, "Scotland": 33, "Thailand": 34, "Yugoslavia": 35, "El-Salvador": 36,
                  "Trinadad&Tobago": 37, "Peru": 38, "Hong": 39, "Holand-Netherlands": 40}

salary_map = {"<=50K": 0, ">50K": 1}


# Helper function to encode data
def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded

# Actually encode adult.data
df["workclass"] = coding(df["workclass"], work_class_map)
df["marital-status"] = coding(df["marital-status"], marital_status_map)
df["occupation"] = coding(df["occupation"], occupation_map)
df["relationship"] = coding(df["relationship"], relationship_map)
df["race"] = coding(df["race"], race_map)
df["sex"] = coding(df["sex"], sex_map)
df["native-country"] = coding(df["native-country"], native_country_map)
df["salary"] = coding(df["salary"], salary_map)

# Filter out rows that aren't numeric (just Education for right now)
df = df._get_numeric_data()

# Save a new encoded version
df.to_csv("adult_encoded.data")



