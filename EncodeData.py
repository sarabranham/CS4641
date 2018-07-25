import pandas as pd
from sklearn.utils import shuffle

# Read in our dataset
# It's sensitive to spaces in the CSV, so no spaces allowed
input_file = "adult.data"
df = pd.read_csv(input_file, header=0)

age_map = {"v0": 0, "v1": 1, "v2": 3, "v3": 3}
fnlwgt_map = {"v0": 0, "v1": 1, "v2": 3, "v3": 3}
education_num_map = {"v0": 0, "v1": 1, "v2": 3, "v3": 3}
capital_gain_map = {"v0": 0, "v3": 3}
capital_loss_map = {"v0": 0, "v3": 3}
hours_per_week_map = {"v0": 0, "v2": 2, "v3": 3}

work_class_map = {"Private": 2, "Self-emp-not-inc": 4, "Self-emp-inc": 7,
                  "Federal-gov": 6, "Local-gov": 5,
                  "State-gov": 3, "Pool": 0,
                  "?": -1}

marital_status_map = {"Married-civ-spouse": 0, "Divorced": 1,
                      "Never-married": 2, "Separated": 3, "Widowed": 4,
                      "Married-spouse-absent": 5, "Pool": 6}

occupation_map = {"Tech-support": 10, "Craft-repair": 8, "Other-service": 1,
                  "Sales": 9, "Exec-managerial": 13,
                  "Prof-specialty": 12, "Handlers-cleaners": 2, "Machine-op-inspct": 5, "Adm-clerical": 6,
                  "Farming-fishing": 4, "Transport-moving": 7, "Priv-house-serv": 0, "Protective-serv": 11,
                  "Armed-Forces": 3, "?": -1}

occupation_map = {"?": -1, "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4,
                  "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8,
                  "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12,
                  "Armed-Forces": 13}

relationship_map = {"Wife": 0, "Own-child": 1, "Husband": 2,
                    "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5}

race_map = {"White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2, "Other": 3, "Black": 4}

sex_map = {"Female": 0, "Male": 1}

native_country_map = {"Canada":0,"Dominican-Republic":1,"Italy":2,"Cuba":3,"Guatemala":4,"China":5,"Germany":6,"Poland":7,"Philippines":8,"Vietnam":9,"South":10,"Jamaica":11,"England":12,"Mexico":13,"El-Salvador":14,"India":15,"Puerto-Rico":16,"United-States":17,"Japan":18,"Taiwan":19,"Pool":20,"Columbia":21}
salary_map = {"'<=50K'": 0, "''>50K'": 1}

# Helper function to encode data
def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded

# Actually encode adult.data
df["age"] = coding(df["age"], age_map)
df["fnlwgt"] = coding(df["fnlwgt"], fnlwgt_map)
df["education-num"] = coding(df["education-num"], education_num_map)
df["capital-gain"] = coding(df["capital-gain"], capital_gain_map)
df["capital-loss"] = coding(df["capital-loss"], capital_loss_map)
df["hours-per-week"] = coding(df["hours-per-week"], hours_per_week_map)
df["workclass"] = coding(df["workclass"], work_class_map)
df["marital-status"] = coding(df["marital-status"], marital_status_map)
df["occupation"] = coding(df["occupation"], occupation_map)
df["relationship"] = coding(df["relationship"], relationship_map)
df["race"] = coding(df["race"], race_map)
df["sex"] = coding(df["sex"], sex_map)
df["native-country"] = coding(df["native-country"], native_country_map)
df["salary"] = coding(df["salary"], salary_map)

# Filter out rows that aren't numeric (just Education for right now)
df = df[['age','workclass','fnlwgt','education-num','marital-status','occupation','relationship','capital-gain','race','capital-loss','hours-per-week','native-country','sex','salary']]
print df.keys()
# Save a new encoded version
df.to_csv("adult_encoded.data", sep=" ", header=None, index=False)

df = shuffle(df)
totalSize = len(df)
folds = 10
testSize = totalSize / folds
df_train = df[0:testSize]
df_test = df[testSize:]

df_train.to_csv("adult_encoded_train.data", sep=" ", header=None, index=False)
df_test.to_csv("adult_encoded_test.data", sep=" ", header=None, index=False)
