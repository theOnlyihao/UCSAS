#%%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pulp
#%%
############### Read in the data ###############
file_path = '/Users/yihao/Documents/Bentley/Courses/Fall 2023/MA707/project/Final Project/data_2022_2023.csv'
gymnastic_data = pd.read_csv(file_path)
gymnastic_data.head()

#%%
def assign_medal_points(rank):
    if rank == 1:
        return 3  # Points for a gold medal
    elif rank == 2:
        return 2  # Points for a silver medal
    elif rank == 3:
        return 1  # Points for a bronze medal
    else:
        return 0  # No points if no medal
    
gymnastic_data['Medal_Points'] = gymnastic_data['Rank'].apply(assign_medal_points)
# %%
# Filter the dataset to only include results from USA gymnasts
usa_gymnastics_data = gymnastic_data[gymnastic_data['Country'] == 'USA']

# We will perform a basic check for duplicates based on gymnast's first and last names, and apparatus, assuming that a gymnast can't have more than one rank for the same apparatus in the same competition and round.
# Any duplicates detected by this criteria will be reviewed.
duplicates_check = usa_gymnastics_data.duplicated(subset=['FirstName', 'LastName', 'Competition', 'Round', 'Apparatus'], keep=False)

# Display potential duplicates if any, and the overall summary of USA gymnasts' performance
potential_duplicates = usa_gymnastics_data[duplicates_check]
summary_of_performance = usa_gymnastics_data.describe()

potential_duplicates, summary_of_performance

#%%
############### EDA ###############
# Distribution of Scores for Team USA gymnasts
plt.figure(figsize=(14, 7))

# Histogram of Scores
plt.subplot(1, 2, 1)
sns.histplot(usa_gymnastics_data['Score'], bins=30, kde=True)
plt.title('Distribution of Scores for Team USA Gymnasts')

# Boxplot of Scores by Apparatus
plt.subplot(1, 2, 2)
sns.boxplot(x='Apparatus', y='Score', data=usa_gymnastics_data, palette='magma')
plt.title('Boxplot of Scores by Apparatus for Team USA Gymnasts')
plt.xticks(rotation=45)
plt.show()
#%%
# Boxplot of Ranks by Apparatus
plt.figure(figsize=(14, 7))
sns.boxplot(x='Apparatus', y='Rank', data=usa_gymnastics_data, palette='magma')
plt.title('Boxplot of Ranks by Apparatus for Team USA Gymnasts')
plt.gca().invert_yaxis()  # Invert y-axis to have the best ranks at the top
plt.xticks(rotation=45)
plt.show()

#%%
# EDA: Distribution of Medal Points for Team USA gymnasts
plt.figure(figsize=(15, 10))
# We will only display the top 20 gymnasts with the most medal points for clarity
top_medal_points = usa_gymnastics_data.groupby(['FirstName', 'LastName'])['Medal_Points'].sum().reset_index()
top_medal_points = top_medal_points.sort_values(by='Medal_Points', ascending=False).head(20)

# Barplot of Medal Points
sns.barplot(x='Medal_Points', y='LastName', data=top_medal_points, hue='FirstName', dodge=False)
plt.title('Top 20 Team USA Gymnasts by Medal Points')
plt.xlabel('Total Medal Points')
plt.ylabel('Gymnasts')
plt.show()

#%%
# Calculate the correlation matrix for the numerical features of the Team USA gymnasts
correlation_matrix = usa_gymnastics_data[['Score', 'D_Score', 'E_Score', 'Rank']].corr()
# Generate a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title('Correlation Heatmap of Scores and Ranks for Team USA Gymnasts')
plt.show()

# %%
# Group the data by gymnast and apparatus to get a summary of their performances
gymnast_performance_summary = usa_gymnastics_data.groupby(['FirstName', 'LastName', 'Apparatus'])\
                                                  .agg({'Rank': ['min', 'mean'], # Minimum and mean rank for medal potential
                                                        'Score': 'mean', # Mean score for performance level
                                                        'Medal_Points': 'sum'}) # Total medal points for weighted medal count
# Flatten the multi-level column index resulted from aggregation
gymnast_performance_summary.columns = ['_'.join(col).strip() for col in gymnast_performance_summary.columns.values]

# Reset index to make 'FirstName', 'LastName', and 'Apparatus' as columns again
gymnast_performance_summary.reset_index(inplace=True)

# We will sort by total medal points to identify top performers, then by mean score as a secondary criterion
gymnast_performance_summary_sorted = gymnast_performance_summary.sort_values(by=['Medal_Points_sum', 'Score_mean'], ascending=[False, False])

# Display top performers based on medal points and scores
gymnast_performance_summary_sorted.head()
# %%
# Identify the top performers for each apparatus based on the historical data
# We will calculate the average score for each gymnast on each apparatus
# Then we will select the top 4 gymnasts based on the average score for the '4 up, 3 count' simulation

# Calculate average score for each gymnast on each apparatus
apparatus_average_scores = usa_gymnastics_data.groupby(['Apparatus', 'FirstName', 'LastName'])\
                                             .agg({'Score': 'mean'})\
                                             .reset_index()

# Sort the results to find the top performers for each apparatus
top_performers_per_apparatus = apparatus_average_scores.sort_values(['Apparatus', 'Score'], ascending=[True, False])

# Now we create a dictionary to hold the top 4 gymnasts for each apparatus
top_performers_dict = {}

for apparatus in top_performers_per_apparatus['Apparatus'].unique():
    top_performers_dict[apparatus] = top_performers_per_apparatus[top_performers_per_apparatus['Apparatus'] == apparatus].head(4)

top_performers_dict['FX']  # Example output for Floor Exercise (FX)

# %%
# Initialize an empty list to hold dataframes for top performers
top_performers_list = []

# Loop through each apparatus and get the top 4 performers
for apparatus in usa_gymnastics_data['Apparatus'].unique():
    top_performers = apparatus_average_scores[apparatus_average_scores['Apparatus'] == apparatus]\
                     .nlargest(4, 'Score')
    top_performers_list.append(top_performers)

# Concatenate all the dataframes in the list into one dataframe
top_performers_all_apparatus = pd.concat(top_performers_list)

def simulate_4_up_3_count(apparatus_data):
    # Sort the gymnasts by score in descending order (best scores first)
    sorted_gymnasts = apparatus_data.sort_values('Score', ascending=False)
    # Take the scores of the top 4 gymnasts
    top_scores = sorted_gymnasts.head(4)['Score']
    # Sum the best 3 scores out of the top 4 for the team score
    team_score = top_scores.nlargest(3).sum()
    return team_score


# the '4 up, 3 count' simulation
team_scores_qualification = top_performers_all_apparatus.groupby('Apparatus')\
                                                        .apply(simulate_4_up_3_count)\
                                                        .reset_index(name='Team_Score')

# Display the team scores for each apparatus
team_scores_qualification.sort_values('Team_Score', ascending=False)

# %%
# For '3 up, 3 count' simulation, we sum the scores for the top three gymnasts for each apparatus
# This function will be used to perform the simulation for each apparatus
def simulate_3_up_3_count(apparatus_data):
    # Sort the gymnasts by score in descending order (best scores first)
    sorted_gymnasts = apparatus_data.sort_values('Score', ascending=False)
    # Take the scores of the top 3 gymnasts since all scores count in the '3 up, 3 count' format
    team_score = sorted_gymnasts.head(3)['Score'].sum()
    return team_score

# Apply the simulation for each apparatus and get the team score
team_scores_final = top_performers_all_apparatus.groupby('Apparatus')\
                                                 .apply(simulate_3_up_3_count)\
                                                 .reset_index(name='Team_Score')

# Display the team scores for each apparatus for the '3 up, 3 count' scenario
team_scores_final.sort_values('Team_Score', ascending=False)

# %%
# Calculate the total medal count for each gymnast
total_medals = usa_gymnastics_data.groupby(['FirstName', 'LastName'])['Medal_Points'].sum().reset_index()

# Calculate the number of gold medals for each gymnast (gold is represented by 3 points in Medal_Points)
gold_medals = usa_gymnastics_data[usa_gymnastics_data['Medal_Points'] == 3].groupby(['FirstName', 'LastName']).size().reset_index(name='Gold_Count')

# Merge the total medals and gold medal counts
medal_counts = total_medals.merge(gold_medals, on=['FirstName', 'LastName'], how='left')
medal_counts['Gold_Count'] = medal_counts['Gold_Count'].fillna(0)

# Calculate the weighted medal count
medal_counts['Weighted_Medal_Count'] = medal_counts['Medal_Points']  # This already represents the weighted count

# Display the gymnasts sorted by total medals, gold medals, and weighted medal count
medal_counts.sort_values(by=['Medal_Points', 'Gold_Count', 'Weighted_Medal_Count'], ascending=False).head()

# %%
# Calculate the average score and standard deviation for each gymnast on each apparatus
gymnast_scores_stats = usa_gymnastics_data.groupby(['FirstName', 'LastName', 'Apparatus'])\
                                         .agg(Average_Score=('Score', 'mean'),
                                              Score_SD=('Score', 'std'),
                                              Routine_Count=('Score', 'count'))\
                                         .reset_index()

# Replace NaN values in Score_SD with 0 (occurs when there's only one score)
gymnast_scores_stats['Score_SD'] = gymnast_scores_stats['Score_SD'].fillna(0)

# Now, let's calculate the number of medals each gymnast has won
# We consider the Medal_Points where 3 points = Gold, 2 points = Silver, 1 point = Bronze
medal_counts = usa_gymnastics_data.groupby(['FirstName', 'LastName'])\
                                 .agg(Total_Medal_Points=('Medal_Points', 'sum'),
                                      Gold_Medal_Count=('Medal_Points', lambda x: (x==3).sum()),
                                      Silver_Medal_Count=('Medal_Points', lambda x: (x==2).sum()),
                                      Bronze_Medal_Count=('Medal_Points', lambda x: (x==1).sum()))\
                                 .reset_index()

# Merge the scores stats with the medal counts to get a combined dataframe
gymnast_performance_summary = pd.merge(gymnast_scores_stats, medal_counts, on=['FirstName', 'LastName'])

# Sort by Average Score and Total Medal Points to identify the top performers
gymnast_performance_summary.sort_values(by=['Average_Score', 'Total_Medal_Points'], ascending=[False, False]).head()

#%%
# Function to select top 3 gymnasts for each apparatus and simulate the '3 up, 3 count' team score
def simulate_team_all_around(top_performers):
    team_all_around_score = 0
    for apparatus in top_performers['Apparatus'].unique():
        apparatus_scores = top_performers[top_performers['Apparatus'] == apparatus]
        team_all_around_score += apparatus_scores.head(3)['Average_Score'].sum()
    return team_all_around_score

# Select top 3 gymnasts for each apparatus based on their average scores
top_3_per_apparatus = gymnast_scores_stats.groupby('Apparatus')\
                                          .apply(lambda x: x.nlargest(3, 'Average_Score'))\
                                          .reset_index(drop=True)

# Simulate the '3 up, 3 count' team score for the all-around event
team_all_around_score = simulate_team_all_around(top_3_per_apparatus)

team_all_around_score
# %%
# We need to identify the threshold scores that typically qualify a gymnast for event finals at the Olympics.
# For the sake of this simulation, let's assume that the top 8 scores in our dataset are a good proxy for finals qualification.

# Define a function to estimate individual event success
def estimate_event_success(apparatus_data, top_n=8):
    # Assuming the top_n scores could be a proxy for finals qualification
    qualifying_scores = apparatus_data.nlargest(top_n, 'Average_Score')['Average_Score'].min()
    # Estimate success by checking how many times gymnasts have scored above this threshold
    apparatus_data['Qualify_Probability'] = apparatus_data['Average_Score'].apply(lambda x: 1 if x >= qualifying_scores else 0)
    return apparatus_data

# Apply the function to each apparatus
individual_event_success = top_3_per_apparatus.groupby('Apparatus').apply(estimate_event_success).reset_index(drop=True)

# Now, let's sort the data to see which gymnasts are most likely to qualify for the finals
individual_event_success = individual_event_success.sort_values(by=['Apparatus', 'Qualify_Probability', 'Average_Score'], ascending=[True, False, False])
individual_event_success[['Apparatus', 'FirstName', 'LastName', 'Average_Score', 'Qualify_Probability']]

# %%
# Data Preparation for ML Model

# Selecting relevant features for the model
# We assume 'Score' is the average score, 'Medal_Points' is the total medal points for each gymnast
features = ['FirstName', 'LastName', 'Apparatus', 'Average_Score', 'Score_SD', 'Total_Medal_Points',
            'Gold_Medal_Count', 'Silver_Medal_Count', 'Bronze_Medal_Count']

# Preparing the dataset for the model, we need to pivot the apparatus to create a feature for each
gymnast_scores_pivot = gymnast_scores_stats.pivot_table(index=['FirstName', 'LastName'], 
                                                        columns='Apparatus', 
                                                        values='Average_Score').reset_index()

# Fill any NaN values with the mean score of each gymnast across all apparatuses
gymnast_scores_pivot = gymnast_scores_pivot.fillna(gymnast_scores_pivot.mean(axis=1, numeric_only=True))
# Merge with the medal counts data
ml_dataset = pd.merge(gymnast_scores_pivot, medal_counts, on=['FirstName', 'LastName'])

# Check the prepared dataset
ml_dataset.head()


# %%
# Dropping non-numeric columns for modeling
ml_features = ml_dataset.drop(['FirstName', 'LastName'], axis=1)

# We need to separate the features and the target variable
X = ml_features.drop('Total_Medal_Points', axis=1)
y = ml_features['Total_Medal_Points']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Recheck for NaN values and fill them with the mean of the columns if necessary
X_train_mean = X_train.mean()
X_train_filled = X_train.fillna(X_train_mean)
X_test_filled = X_test.fillna(X_train_mean)

# Normalizing the features again
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filled)
X_test_scaled = scaler.transform(X_test_filled)
model = RandomForestRegressor(n_estimators=100, random_state=42)
# Retraining the RandomForestRegressor model
model.fit(X_train_scaled, y_train)

# Redoing the prediction on the testing set
y_pred = model.predict(X_test_scaled)

# Recalculating the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
mse

# %%
# Predicting the total medal points for each gymnast using the trained model on the entire dataset
X_filled = X.fillna(X_train_mean)
X_scaled = scaler.transform(X_filled)  # Normalize the entire feature set
predicted_medal_points = model.predict(X_scaled)

# Adding the predictions back to the original dataset to see actual vs predicted
ml_dataset['Predicted_Medal_Points'] = predicted_medal_points

# Now we sort the dataset by the predicted medal points to see the top predicted medalists
top_predicted_medalists = ml_dataset.sort_values(by='Predicted_Medal_Points', ascending=False)

# Show the top predicted medalists along with their actual total medal points for comparison
top_predicted_medalists[['FirstName', 'LastName', 'Total_Medal_Points', 'Predicted_Medal_Points']].head()
# %%
# Initialize the optimization problem
team_selection = pulp.LpProblem("Team_Selection", pulp.LpMaximize)

# Create decision variables for each gymnast
gymnast_vars = [pulp.LpVariable(f'gymnast_{i}', cat='Binary') for i in range(len(ml_dataset))]

# Objective function: Maximize the sum of predicted medal points
team_selection += pulp.lpSum([gymnast_vars[i] * ml_dataset.loc[i, 'Predicted_Medal_Points'] for i in range(len(ml_dataset))])

# Constraint: Select exactly 5 gymnasts
team_selection += pulp.lpSum(gymnast_vars) == 5, "Select_5_gymnasts"

# Solve the problem
team_selection.solve()

# Extract the results: The selected gymnasts will have their corresponding decision variables set to 1
selected_gymnasts_indices = [i for i in range(len(ml_dataset)) if pulp.value(gymnast_vars[i]) == 1]
selected_gymnasts = ml_dataset.iloc[selected_gymnasts_indices]

# Display the selected gymnasts
print(selected_gymnasts[['FirstName', 'LastName', 'Predicted_Medal_Points']])

selected_gymnasts_df = ml_dataset.iloc[selected_gymnasts_indices]
selected_gymnasts_df = selected_gymnasts_df[['FirstName', 'LastName', 'Predicted_Medal_Points']]

# Display the DataFrame
selected_gymnasts_df

# %%
