import pandas as pd
from sklearn import tree
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import numpy as np

# Check if processed data already exists
if os.path.exists('processed_cdc_data.csv'):
    print("Loading saved processed data...")
    df = pd.read_csv('processed_cdc_data.csv')
    print(f"Loaded shape: {df.shape}")
else:
    #Process Data
    
    #Instance Attritubes
    used_columns = [
        "age_group",
        "sex",
        "race_ethnicity_combined",
        "hosp_yn",
        "medcond_yn",
        "death_yn" # target variable
    ]

    url_discover = "https://data.cdc.gov/resource/vbim-akqf.csv?$select=age_group&$group=age_group"
    age_groups_df = pd.read_csv(url_discover)
    age_groups = age_groups_df['age_group'].dropna().tolist()

    dfs = []

    #Sample 1,000,000 instances from each age group
    for age in age_groups:
        if age == "Missing" or pd.isna(age):
            continue  # Skip missing/NA age groups
        
        url = f"https://data.cdc.gov/resource/vbim-akqf.csv?age_group={age.replace(' ', '%20').replace('+', '%2B')}&$limit=1000000"
        temp_df = pd.read_csv(url, usecols=used_columns)
        dfs.append(temp_df)
        print(f"  Loaded {len(temp_df):,} rows for '{age}'")

    df = pd.concat(dfs, ignore_index=True)

    #Display basic info about dataset (before cleaning)
    print("\ndataset loaded successfully")
    print("Shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\n####################################################################################################################################")

    #Cleaning the dataset

    #Standardize the values for easy removal
    cleanup = {
        "Yes" : "Yes",
        "No"  : "No",
        "Unknown" : pd.NA,
        "Missing" : pd.NA,
        "NA" : pd.NA,
        "Not stated" : pd.NA,
        None : pd.NA
    }

    #New age groups: 0-19, 20-39, 40-59, 60-79, 80+
    age_mapping = {
        "0 - 9 Years" : "0-19",
        "10 - 19 Years" : "0-19",
        "20 - 29 Years" : "20-39",
        "30 - 39 Years" : "20-39",
        "40 - 49 Years" : "40-59",
        "50 - 59 Years" : "40-59",
        "60 - 69 Years" : "60-79",
        "70 - 79 Years" : "60-79",
        "80+ Years" : "80+"
    }

    #New race/ethnicity groups; removes "non-hispanic" from each
    race_ethnicity_mapping = {
        "American Indian/Alaska Native, Non-Hispanic" : "American Indian/Alaska Native",
        "Asian, Non-Hispanic" : "Asian",
        "Black, Non-Hispanic" : "Black",
        "Multiple/Other, Non-Hispanic" : "Multiple/Other",
        "Native Hawaiian/Other Pacific Islander, Non-Hispanic" : "Native Hawaiian/Other Pacific Islander",
        "White, Non-Hispanic" : "White",
    }

    #Map new age and race_ethnicity groups
    df["age_group"] = df["age_group"].replace(age_mapping)
    df["race_ethnicity_combined"] = df["race_ethnicity_combined"].replace(race_ethnicity_mapping)

    #Clean up the messy categorical columns
    for col in ["hosp_yn", "medcond_yn", "death_yn", "sex", "race_ethnicity_combined"]:
        df[col] = df[col].replace(cleanup)

    #Drop instances with unknown/missing attributes
    df = df[df['death_yn'] != -1]
    df = df.dropna()

    #Convert the target variable to int
    df["death_yn"] = df["death_yn"].map({"Yes": 1, "No": 0, "Unknown": -1})

    '''Count of each attribute
    print(f"Age Group Counts: {df['age_group'].value_counts()}\n")
    print(f"Sex Counts: {df['sex'].value_counts()}\n")
    print(f"Race/Ethnicity Counts: {df['race_ethnicity_combined'].value_counts()}\n")
    print(f"Hospitalizaion Counts: {df['hosp_yn'].value_counts()}\n")
    print(f"MedCond Counts: {df['medcond_yn'].value_counts()}\n")
    print(f"Death Counts: {df['death_yn'].value_counts()}")
    '''

    #One-hot encode the categorical variables into its own binary column
    df = pd.get_dummies(df, columns=["age_group", "sex", "race_ethnicity_combined", "hosp_yn", "medcond_yn"], drop_first=False)

    #Display basic info about dataset (after cleaning)
    print("\nCleaning complete!")
    print("New shape:", df.shape)

    print("\nFirst 5 cleaned rows:")
    print(df.head())

    print("\nFinal column count:", len(df.columns))
    print("\nColumn names:")
    print(df.columns.tolist())
    
    #Save the processed data
    df.to_csv('processed_cdc_data.csv', index=False)
    print("\nSaved processed data to 'processed_cdc_data.csv'")






#Build Gini Index Decision Tree
print("Building Gini Index Decision Tree\n")

#Data Splitting (Train, Validation, Test)
X = df.drop('death_yn', axis=1)
y = df['death_yn']


#Split into temp (80%) and Test (20%) - Using stratification for class balance
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

#Split temp (80%) into Training (60% of total) and Validation (20% of total)
#(25% of the temporary set is 20% of the total set)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0, stratify=y_temp)

print(f"\nData Split:")
print(f"Training set size: {len(X_train)} ({len(X_train)/len(df):.0%})")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(df):.0%})")
print(f"Testing set size: {len(X_test)} ({len(X_test)/len(df):.0%})")

#Fit the unpruned tree on the training set
clf_unpruned = tree.DecisionTreeClassifier(random_state=0)
clf_unpruned.fit(X_train, y_train)

#Calculate the effective alphas and impurities for the path
path = clf_unpruned.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    #Build a tree for each alpha value
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

#Remove the last tree (which is the single-node root)
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

#Compute the F1-score over the validation dataset for model selection
val_f1_scores = [f1_score(y_val, clf.predict(X_val), average='weighted') for clf in clfs]
train_f1_scores = [f1_score(y_train, clf.predict(X_train), average='weighted') for clf in clfs]

#Find the index of the model with the highest F1-score on the VALIDATION set
best_clf_index = np.argmax(val_f1_scores)
best_pruned_clf = clfs[best_clf_index]
optimal_alpha = ccp_alphas[best_clf_index]
max_val_f1 = val_f1_scores[best_clf_index]

print("\nModel Selection Results")
print(f"Optimal ccp_alpha selected using Validation set F1-Score: {optimal_alpha:.6f}")
print(f"Validation Set F1-Score at optimal alpha: {max_val_f1:.4f}")
print(f"Node count of best pruned tree: {best_pruned_clf.tree_.node_count}")

y_pred_test = best_pruned_clf.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_test)
final_f1 = f1_score(y_test, y_pred_test, average='weighted')

print("\nEvaluation on Testing Set")
print(f"Final Test Set Accuracy: {final_accuracy:.4f}")
print(f"Final Test Set F1-Score (Weighted): {final_f1:.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['No Death', 'Death']))

#Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix (Test Set):")
#[[True Negatives, False Positives]
# [False Negatives, True Positives]]
print(cm)


#Visualization of the Best Pruned Tree
#Prepare dot file for graphviz (Pruned Tree)
dot_data_pruned = tree.export_graphviz(
    best_pruned_clf, out_file=None, 
    feature_names=X.columns.tolist(), 
    class_names=['No Death', 'Death'], 
    filled=True, rounded=True, 
    special_characters=True,
    #max_depth=5 # Limit depth for readability of the final visualization
) 
graph_pruned = graphviz.Source(dot_data_pruned) 
graph_pruned.render("decision_tree_pruned_best", format="png", cleanup=True)

#Visualization of the Original Unpruned Tree (For Comparison)
#Prepare dot file for graphviz (Unpruned Tree)
dot_data_original = tree.export_graphviz(
    clf_unpruned, out_file=None, 
    feature_names=X.columns.tolist(),
    class_names=['No Death', 'Death'], 
    filled=True, rounded=True, 
    special_characters=True
) 
graph_original = graphviz.Source(dot_data_original) 
graph_original.render("decision_tree_original_unpruned", format="png", cleanup=True)





#Build Naive Bayes Classifier
print("\nBuilding Naive Bayes Classifer\n")

X = df.drop("death_yn", axis=1)
y = df["death_yn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
y_test = y_test.astype(int)
print(sorted(y_test.unique()))
print(y_test.dtype)
bayes = BernoulliNB(alpha = 0.1)

bayes.fit(X_train, y_train)

y_pred = bayes.predict(X_test)
y_pred = y_pred.astype(int)
print("Model classes:", bayes.classes_)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['No Death', 'Death']))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig1, ax = plt.subplots()
cax = ax.matshow(cm, cmap="viridis")
plt.colorbar(cax)

#Set tick labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

#[[True Negatives, False Positives]
# [False Negatives, True Positives]]
ax.set_xticklabels(["True Negatives", "False Positives"])
ax.set_yticklabels(["False Negatives", "True Positives"])

#Annotate cells
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, str(val), ha='center', va='center', color='black')

#Label axes
ax.set_xlabel("Actual Label")
ax.xaxis.set_label_position('top')
ax.set_ylabel("Predicted Label")

# Display Matrix
plt.show()

#Figure 2 Bar Plot based of Feature Importance
#Compute feature importance
feature_importance = bayes.feature_log_prob_[1] - bayes.feature_log_prob_[0]

#Put into DataFrame and sort by importance
feat_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": feature_importance
}).sort_values(by="importance", ascending=False)

#Map feature names to readable versions
#fill unmapped features with original names
feature_name_map = {
    "hosp_yn_Yes": "Hospitalization=Yes",
    "age_group_80+": "Age=80+",
    "medcond_yn_Yes": "Preexisting Condition=Yes",
    "race_ethnicity_combined_Asian": "Ethnicity=Asian",
    "sex_Female": "Sex=Female",
    "race_ethnicity_combined_Hispanic/Latino": "Ethnicity=Hispanic/Latino",
    "age_group_60-79": "60<=Age<=79",
}
feat_df['readable_name'] = feat_df['feature'].map(feature_name_map).fillna(feat_df['feature'])

#Select top 7 features
top_feats = feat_df[:7]

#Create vertical bar plot
fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.bar(top_feats['readable_name'], top_feats['importance'], color='teal')

#Labels and title
ax2.set_ylabel("Log Prob Difference (Class 1 - Class 0)")
ax2.set_xlabel("Feature")
ax2.set_title("Top 7 Most Decisive Features")

#Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()  #ensure everything fits
plt.show()