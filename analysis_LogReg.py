import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
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

# ===================================================================
# Logistic Regression Model No Undersampling - ALL FEATURES
# ===================================================================

X1 = df.drop("death_yn", axis=1)
y1 = df["death_yn"]
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=0
)
#Create Scalars
scaler1 = StandardScaler()
X1_train = scaler1.fit_transform(X1_train)
X1_test = scaler1.transform(X1_test)
#Create Model
log_reg1 = LogisticRegression(class_weight='balanced',max_iter=2500)
#Train Model
log_reg1.fit(X1_train, y1_train)
#Make Predictions
y1_pred = log_reg1.predict(X1_test)
y1_prob = log_reg1.predict_proba(X1_test)[:, 1]

# ===================================================================
# Logistic Regression Model No Undersampling - TOP 7 FEATURES
# ===================================================================

X2 = df[["hosp_yn_Yes","age_group_80+","medcond_yn_Yes","race_ethnicity_combined_Asian",
         "sex_Female","race_ethnicity_combined_Hispanic/Latino","age_group_60-79"]]
y2 = df["death_yn"]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=0
)
#Create Scalars
scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)
X2_test = scaler2.transform(X2_test)
#Create Model
log_reg2 = LogisticRegression(class_weight='balanced',max_iter=2500)
#Train Model
log_reg2.fit(X2_train, y2_train)
#Make Predictions
y2_pred = log_reg2.predict(X2_test)
y2_prob = log_reg2.predict_proba(X2_test)[:, 1]

# ===================================================================
# Logistic Regression with 70:30 Undersampling — ALL FEATURES
# ===================================================================

X3 = df.drop("death_yn", axis=1)
y3 = df["death_yn"]

# Count minority class (1)
minority_count = sum(y3 == 1)

# For 70:30 ratio:
desired_no = int((0.70 / 0.30) * minority_count)

rus_custom = RandomUnderSampler(
    sampling_strategy={0: desired_no, 1: minority_count},
    random_state=0
)

# Apply undersampling BEFORE train/test split
X3_res, y3_res = rus_custom.fit_resample(X3, y3)

# Split
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3_res, y3_res, test_size=0.2, random_state=0
)

# Scale
scaler3 = StandardScaler()
X3_train = scaler3.fit_transform(X3_train)
X3_test = scaler3.transform(X3_test)

# Model
log_reg3 = LogisticRegression(max_iter=2500)
log_reg3.fit(X3_train, y3_train)

# Predictions
y3_pred = log_reg3.predict(X3_test)

X4 = df[[
    "hosp_yn_Yes","age_group_80+","medcond_yn_Yes",
    "race_ethnicity_combined_Asian","sex_Female",
    "race_ethnicity_combined_Hispanic/Latino","age_group_60-79"
]]
y4 = df["death_yn"]

# Count minority class (1)
minority_count = sum(y4 == 1)

# 70:30 ratio target
desired_no = int((0.70 / 0.30) * minority_count)

rus_custom2 = RandomUnderSampler(
    sampling_strategy={0: desired_no, 1: minority_count},
    random_state=0
)

# Apply undersampling
X4_res, y4_res = rus_custom2.fit_resample(X4, y4)

# Split
X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4_res, y4_res, test_size=0.2, random_state=0
)

# Scale
scaler4 = StandardScaler()
X4_train = scaler4.fit_transform(X4_train)
X4_test = scaler4.transform(X4_test)

# Model
log_reg4 = LogisticRegression(max_iter=2500)
log_reg4.fit(X4_train, y4_train)

# Predictions
y4_pred = log_reg4.predict(X4_test)

# ===================================================================
# Comparison Visualizations of Four Models
# ===================================================================

# -------------------------------------------------------------------
# 1.) Classification Report Break Down
# -------------------------------------------------------------------
models = {
    "A: All 20 (no undersample)": (log_reg1, X1_test, y1_test, scaler1),
    "B: Top 7 (no undersample)": (log_reg2, X2_test, y2_test, scaler2),
    "C: All 20 (70:30 undersample)": (log_reg3, X3_test, y3_test, scaler3),
    "D: Top 7 (70:30 undersample)": (log_reg4, X4_test, y4_test, scaler4)
}

metrics_list = []

for name, (model, X_test, y_test, scaler) in models.items():

    if scaler is not None:
        X_test_df = pd.DataFrame(X_test, columns=scaler.feature_names_in_)
        X_scaled = scaler.transform(X_test_df)
    else:
        X_scaled = X_test

    y_pred = model.predict(X_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)

    keys = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    neg, pos = keys[0], keys[1]

    metrics_list.append({
        "Precision_No": report[neg]["precision"],
        "Recall_No": report[neg]["recall"],
        "F1_No": report[neg]["f1-score"],
        "Precision_Yes": report[pos]["precision"],
        "Recall_Yes": report[pos]["recall"],
        "F1_Yes": report[pos]["f1-score"]
    })

df_metrics = pd.DataFrame(metrics_list)
df_metrics.index = list(models.keys())

df_display = pd.DataFrame({
    "Model": df_metrics.index,
    "No Death-Class Metrics": df_metrics[["Precision_No", "Recall_No", "F1_No"]]
        .apply(lambda row: f"Precision = {row.iloc[0]:.3f}\nRecall = {row.iloc[1]:.3f}\nF1 Score = {row.iloc[2]:.3f}", axis=1),
    "Death-Class Metrics": df_metrics[["Precision_Yes", "Recall_Yes", "F1_Yes"]]
        .apply(lambda row: f"Precision = {row.iloc[0]:.3f}\nRecall = {row.iloc[1]:.3f}\nF1 Score = {row.iloc[2]:.3f}", axis=1)
})

fig, ax = plt.subplots(figsize=(9, 6))
ax.axis("off")

col_widths = [0.5, 0.25, 0.25]

table = ax.table(
    cellText=df_display.values,
    colLabels=df_display.columns,
    cellLoc='center',
    loc='center',
    colWidths=col_widths
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 3.5)

# Set monospace font
for (row, col), cell in table.get_celld().items():
    cell.get_text().set_fontfamily("monospace")

# Center title over entire table
plt.suptitle("Model Performance Summary", fontsize=18, y=0.85)

plt.savefig("LogReg_EvaluationSummary", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------
# 2.) Confusion Matrix Comparison
# -------------------------------------------------------------------

model_names = [
    "A: All 20 (no undersample)",
    "B: Top 7 (no undersample)",
    "C: All 20 (70:30 undersample)",
    "D: Top 7 (70:30 undersample)"
]

y_tests = [y1_test, y2_test, y3_test, y4_test]
y_preds = [y1_pred, y2_pred, y3_pred, y4_pred]

fig, axs = plt.subplots(2, 2, figsize=(9, 11))
axs = axs.ravel()

for i, ax in enumerate(axs):
    cm = confusion_matrix(y_tests[i], y_preds[i])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(ax=ax, colorbar=True)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # X-label at top
    ax.set_xlabel("Actual Label", fontsize=9)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Y-label on left
    ax.set_ylabel("Predicted Label", fontsize=9)
    ax.yaxis.set_label_position("left")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Actual Positive", "Actual Negative"], fontsize=7)
    ax.set_yticklabels(["Predicted Positive", "Predicted Negative"], fontsize=7)

    ax.set_title(model_names[i], fontsize=10)
    ax.title.set_position([0.5, -0.25])

plt.tight_layout()
plt.savefig("LogReg_HeatMaps.png", dpi=300, bbox_inches="tight")
plt.show()
