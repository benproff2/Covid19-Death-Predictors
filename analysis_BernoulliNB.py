import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import os

# Check if processed data already exists
if os.path.exists('processed_cdc_data.csv'):
    print("Loading saved processed data...")
    df = pd.read_csv('processed_cdc_data.csv')
    print(f"Loaded shape: {df.shape}")
else:
    print("Processing fresh data from CDC...")

    # LOAD CDC DATASET
    used_columns = [
        "age_group",
        "sex",
        "race_ethnicity_combined",
        "hosp_yn",
        "medcond_yn",
        "death_yn"  # target variable
    ]

    # print("Discovering all age groups in dataset...")
    url_discover = "https://data.cdc.gov/resource/vbim-akqf.csv?$select=age_group&$group=age_group"
    age_groups_df = pd.read_csv(url_discover)
    age_groups = age_groups_df['age_group'].dropna().tolist()

    print(age_groups)

    dfs = []

    # Samples 1,000,000 from each age group
    for age in age_groups:
        if age == "Missing" or pd.isna(age):
            continue  # Skip missing age groups

        url = f"https://data.cdc.gov/resource/vbim-akqf.csv?age_group={age.replace(' ', '%20').replace('+', '%2B')}&$limit=1000000"
        temp_df = pd.read_csv(url, usecols=used_columns)
        dfs.append(temp_df)
        print(f"  Loaded {len(temp_df):,} rows for '{age}'")

    df = pd.concat(dfs, ignore_index=True)

    print(f"\nFinal balanced dataset shape: {df.shape}")
    print("\nAge group distribution:")
    print(df['age_group'].value_counts().sort_index())

    # DISPLAY BASIC INFORMATION ABOUT THE DATASET (before cleaning)
    print("\ndataset loaded successfully")
    print("Shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print(
        "\n####################################################################################################################################")

    # DISPLAY THE DATA WITH CLEANING INVOLVED
    print("\ncleaning the dataset...")

    # standardize the values
    cleanup = {
        "Yes": "Yes",
        "No": "No",
        "Unknown": pd.NA,
        "Missing": pd.NA,
        "NA": pd.NA,
        "Not stated": pd.NA,
        None: pd.NA
    }

    age_mapping = {
        "0 - 9 Years": "0-19",
        "10 - 19 Years": "0-19",
        "20 - 29 Years": "20-39",
        "30 - 39 Years": "20-39",
        "40 - 49 Years": "40-59",
        "50 - 59 Years": "40-59",
        "60 - 69 Years": "60-79",
        "70 - 79 Years": "60-79",
        "80+ Years": "80+"
    }

    race_ethnicity_mapping = {
        "American Indian/Alaska Native, Non-Hispanic": "American Indian/Alaska Native",
        "Asian, Non-Hispanic": "Asian",
        "Black, Non-Hispanic": "Black",
        "Multiple/Other, Non-Hispanic": "Multiple/Other",
        "Native Hawaiian/Other Pacific Islander, Non-Hispanic": "Native Hawaiian/Other Pacific Islander",
        "White, Non-Hispanic": "White",
    }

    df["age_group"] = df["age_group"].replace(age_mapping)
    df["race_ethnicity_combined"] = df["race_ethnicity_combined"].replace(race_ethnicity_mapping)

    # clean up the messy categorical columns
    for col in ["hosp_yn", "medcond_yn", "death_yn", "sex", "race_ethnicity_combined"]:
        df[col] = df[col].replace(cleanup)

    # convert the target variable to int
    df["death_yn"] = df["death_yn"].map({"Yes": 1, "No": 0, "Unknown": -1})

    # Drop unknowns
    df = df[df['death_yn'] != -1]
    df = df.dropna()

    print(f"Age Group Counts: {df['age_group'].value_counts()}\n")
    print(f"Sex Counts: {df['sex'].value_counts()}\n")
    print(f"Race/Ethnicity Counts: {df['race_ethnicity_combined'].value_counts()}\n")
    print(f"Hospitalizaion Counts: {df['hosp_yn'].value_counts()}\n")
    print(f"MedCond Counts: {df['medcond_yn'].value_counts()}\n")
    print(f"Death Counts: {df['death_yn'].value_counts()}")

    # one-hot encode the categorical variables into its own binary column
    df = pd.get_dummies(df, columns=["age_group", "sex", "race_ethnicity_combined", "hosp_yn", "medcond_yn"],
                        drop_first=False)

    # DISPLAY BASIC INFORMATION ABOUT THE DATASET (after cleaning)
    print("\nCleaning complete!")
    print("New shape:", df.shape)

    print("\nFirst 5 cleaned rows:")
    print(df.head())

    print("\nFinal column count:", len(df.columns))
    print("\nColumn names:")
    print(df.columns.tolist())

    # Save the processed data
    df.to_csv('processed_cdc_data.csv', index=False)
    print("\nSaved processed data to 'processed_cdc_data.csv'")

# Bayes Classifier
print(f"{df.shape}\n")

X = df.drop("death_yn", axis=1)
y = df["death_yn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
y_test = y_test.astype(int)
print(sorted(y_test.unique()))
print(y_test.dtype)
bayes = BernoulliNB(alpha = 0.5)

bayes.fit(X_train, y_train)

y_pred = bayes.predict(X_test)
y_pred = y_pred.astype(int)
print("Model classes:", bayes.classes_)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#--------------------------
# Figure 1 Confusion Matrix
#--------------------------
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
fig1, ax = plt.subplots()
cax = ax.matshow(cm, cmap="viridis")
plt.colorbar(cax)
# Set tick labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Actually Positive", "Actually Negative"])
ax.set_yticklabels(["Predicted Positive", "Predicted Negative"])
# Annotate cells
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, str(val), ha='center', va='center', color='black')
# Label axes
ax.set_xlabel("Actual Label")
ax.xaxis.set_label_position('top')
ax.set_ylabel("Predicted Label")
# Display Matrix
plt.savefig("BernoulliNB_HeatMap.png", dpi=300, bbox_inches="tight")
plt.show()

#----------------------------------------------
# Figure 2 Bar Plot based of Feature Importance
#----------------------------------------------
# Compute feature importance
feature_importance = bayes.feature_log_prob_[1] - bayes.feature_log_prob_[0]

# Put into DataFrame and sort by importance
feat_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": feature_importance
}).sort_values(by="importance", ascending=False)

# Map feature names to readable versions; fill unmapped features with original names
feature_name_map = {
    "hosp_yn_Yes": "Hospitalization=Yes",
    "age_group_80+": "Age=80+",
    "medcond_yn_Yes": "Preexisting Condition=Yes",
    "race_ethnicity_combined_Asian": "Ethnicity=Asian",
    "sex_Female": "Sex=Female",
    "race_ethnicity_combined_Hispanic/Latino": "Ethnicity=Hispanic/Latino",
    "age_group_60-79": "60<Age<79",
}
feat_df['readable_name'] = feat_df['feature'].map(feature_name_map).fillna(feat_df['feature'])

# Select top 7 features
top_feats = feat_df[:7]

# Create vertical bar plot
fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.bar(top_feats['readable_name'], top_feats['importance'], color='teal')

# Labels and title
ax2.set_ylabel("Log Prob Difference (Class 1 - Class 0)")
ax2.set_xlabel("Feature")
ax2.set_title("Top 7 Most Decisive Features")

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()  # ensure everything fits
plt.savefig("BernoulliNB_TopSevenFeatures.png", dpi=300, bbox_inches="tight")
plt.show()