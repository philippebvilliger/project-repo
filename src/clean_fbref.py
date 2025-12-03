import pandas as pd  # import pandas for data manipulation
import numpy as np  # import numpy for numerical operations e.g. NaN handling

# ============================================

print("="*60)
print("CLEANING FBREF DATA")
print("="*60)

# Load the raw combined FBref data
input_file = 'data/processed/fbref_stats_raw.csv'  # path to the raw FBref data that has all raw stats combined
print(f"\n Loading data from: {input_file}")

try:
    df = pd.read_csv(input_file)  # read the CSV file into a pandas DataFrame
    print(f"Loaded {len(df)} records")
except FileNotFoundError: # exception handling if file not found
    print(f" Error: File not found at {input_file}")
    print("   Make sure you ran combine_fbref_files.py first!") # you must have executed this scipt first in order to have the combined raw data
    exit(1)

print(f"\n Original columns: {len(df.columns)}")
print(f" Sample columns: {df.columns.tolist()[:10]}") # displays the first 10 columns as a sample

# ============================================


print("\n" + "-"*60)
print("STEP 1: Removing Unnamed columns")
print("-"*60)
# There are often columns named 'Unnamed: x' that are empty; we have to remove these
unnamed_cols = [col for col in df.columns if 'Unnamed' in col] # we create a list with all the Unnamed columns
print(f"   Found {len(unnamed_cols)} Unnamed columns")

df = df.drop(columns=unnamed_cols) # we delete these columns from the dataframe
print(f" Removed Unnamed columns. Now have {len(df.columns)} columns")

# ============================================


print("\n" + "-"*60)
print("STEP 2: Selecting essential columns")
print("-"*60)

# We make a list with all the essential columns we want to keep
# We'll handle duplicates (like Gls, Gls.1) by taking the first occurrence
essential_cols = [
    'Player',      # Player name
    'Nation',      # Nationality
    'Pos',         # Position
    'Squad',       # Team/Club
    'Age',         # Age
    'Born',        # Birth year
    'MP',          # Matches Played
    'Starts',      # Games Started
    'Min',         # Minutes Played
    '90s',         # 90s played (minutes/90)
    'Gls',         # Goals
    'Ast',         # Assists
    'G+A',         # Goals + Assists
    'G-PK',        # Goals excluding penalties
    'PK',          # Penalty kicks made
    'PKatt',       # Penalty kicks attempted
    'CrdY',        # Yellow cards
    'CrdR',        # Red cards
    'xG',          # Expected Goals
    'npxG',        # Non-penalty Expected Goals
    'xAG',         # Expected Assisted Goals
    'npxG+xAG',    # Non-penalty Expected Goals + Expected Assisted Goals
    'PrgC',        # Progressive Carries
    'PrgP',        # Progressive Passes
    'PrgR',        # Progressive Receptions
    'season',      # Season
    'league'       # League
]

# We make a list of the columns that are actually in the dataframe
available_cols = []
for col in essential_cols: # for each of the essential columns defined above
    # Sometimes when you merge csv files, column names get duplicated e.g. Gls, Gls.1
    matching_cols = [c for c in df.columns if c == col or c.startswith(col + '.')] # We make a list with all matching columns
    # Say the col is 'Gls' then we look for 'Gls' or 'Gls.1', 'Gls.2' because it starts with 'Gls.'
    if matching_cols: # if these exist
        available_cols.append(matching_cols[0])  # we add only the first occurrence to available_cols
    elif col in df.columns: # else we just add the column that don't have duplicates
        available_cols.append(col)

print(f"   Found {len(available_cols)} out of {len(essential_cols)} essential columns")
print(f"   Columns to keep: {available_cols}")

df_clean = df[available_cols].copy() 
# you want to make a copy of the dataframe with these newly selected available columns
# You want to have the original dataframe intact for reference and a new cleaned dataframe with only the essential columns

# Rename any columns with .1, .2 suffixes back to original names
rename_dict = {}
for col in df_clean.columns: # for each column in the new cleaned dataframe
    if '.' in col and col.split('.')[0] in essential_cols: # we only keep the first part of the name before the dot e.g. Gls in Gls.1
        original_name = col.split('.')[0]
        rename_dict[col] = original_name # we add this to a dictionary to rename later with the current column name as key and original name as value

if rename_dict: # if it's not empty
    print(f"\n   Renaming columns: {rename_dict}") # we print the renaming dictionary
    df_clean = df_clean.rename(columns=rename_dict) # If in the new cleaned dataframe there are columns with .1, .2 suffixes we rename them back to original names
    # remember that we only kept the first occurrence of these columns earlier so could be Gls.1 but we renamed it back to Gls

print(f" Selected essential columns. Now have {len(df_clean.columns)} columns")

# ============================================


print("\n" + "-"*60)
print("STEP 3: Cleaning data types and values")
print("-"*60)

# Remove rows where Player is null or empty
before_count = len(df_clean) # length of cleaned dataframe before removing rows with missing player names
df_clean = df_clean[df_clean['Player'].notna()] # .notna() returns True for non-missing values i.e., those we want to keep
# so this line only keeps rows where values in 'Player' column are not missing (NaN)
df_clean = df_clean[df_clean['Player'] != ''] # this removes any rows where Player is an empty string
after_count = len(df_clean) # length of cleaned dataframe after removing rows with missing player names
if before_count != after_count: # if any removals actually occurred
    print(f"   Removed {before_count - after_count} rows with missing player names")

# These are the columns that should have numeric values
numeric_cols = ['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A', 
                'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR']

#
if 'xG' in df_clean.columns: # if there are expected statistics in the cleaned dataframe, we want to include these as well in the numeric columns
    numeric_cols.extend(['xG', 'npxG', 'xAG', 'npxG+xAG']) #.extend() adds multiple items to the list
if 'PrgC' in df_clean.columns: # if there are progressive statistics in the cleaned dataframe, we want to include these as well in the numeric columns
    numeric_cols.extend(['PrgC', 'PrgP', 'PrgR']) 
# Only keep columns that are actually in the cleaned dataframe
numeric_cols = [col for col in numeric_cols if col in df_clean.columns]

print(f"   Converting {len(numeric_cols)} columns to numeric")

for col in numeric_cols:
    # Replace commas with nothing (for numbers like "1,234")
    if df_clean[col].dtype == 'object': #.dytype checks the data type of the column. # object dtype usually means string 
        df_clean[col] = df_clean[col].astype(str).str.replace(',', '').str.replace(' ', '') #astype(str) converts the column to string type
        # We convert it into a string just in case there are any non-string values that would cause errors
    # Convert to numeric, replacing errors with NaN
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')  # we are converting the cleaned column 
    # to_numeric is a pandas function that converts a column to numbers like integers or floats
    # errors='coerce' means that if we can't convert them into numbers, we set them to NaN

    # Turn the missing values into 0 for all columns except Age
    if col != 'Age':
        df_clean[col] = df_clean[col].fillna(0)
        
# we check for any remaining missing values in Age and remove those rows    
df_clean = df_clean[df_clean['Age'].notna()]  # df_clean[] only keeps rows where the condition inside is True
# so here we only keep rows where Age is not missing (not NaN)

# Convert Age to integer
df_clean['Age'] = df_clean['Age'].astype(int)

print(" Converted numeric columns and cleaned values")

# ============================================

print("\n" + "-"*60)
print("STEP 4: Standardizing player names")
print("-"*60)

# We convert the player column into a string and remove any whitespace thanks to stip()
df_clean['Player'] = df_clean['Player'].str.strip()

# Remove any special characters that might cause matching issues
# But keep accents (é, ñ, etc.) as they're part of names
df_clean['Player'] = df_clean['Player'].str.replace(r'\s+', ' ', regex=True)
# r'\s+' is a regex pattern that matches any sequence of whitespace like spaces 
# regex =True tells pandas that we're using a regex pattern
# so here we're replacing multiple spaces with a single space i.e., ' '

print(" Standardized player names")

# ============================================

print("\n" + "-"*60)
print("STEP 5: Removing duplicates")
print("-"*60)

before_dedup = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['Player', 'Squad', 'season', 'league'], keep='first') #
after_dedup = len(df_clean)
# drop_duplicates removes duplicate rows form the dataframe
# Here the duplicates are defined as rows where the combination of Player, Squad, season, and league are the same
# keep='first' means that we keep the first of a set of duplicates and remove the rest

if before_dedup != after_dedup: # if we actually removed any duplicates
    print(f"   Removed {before_dedup - after_dedup} duplicate records")
else: # if we didn't remove any duplicates
    print("   No duplicates found")

# ============================================


print("\n" + "-"*60)
print("STEP 6: Creating derived columns")
print("-"*60)

# 90s means how many full 90-minute matches a player has played
if 'Gls' in df_clean.columns and '90s' in df_clean.columns: 
    df_clean['Gls_per_90'] = np.where(df_clean['90s'] > 0, # we create a new column Gls_per_90 thanks to the previous two columns 
                                       df_clean['Gls'] / df_clean['90s'], 
                                       0) # where() is a numpy function that works like an if-else statement
                                       # 0 if  the player has played less than a full 90-minute match

# This is the same principle for assists per 90 minute matches
if 'Ast' in df_clean.columns and '90s' in df_clean.columns:
    df_clean['Ast_per_90'] = np.where(df_clean['90s'] > 0, 
                                       df_clean['Ast'] / df_clean['90s'], 
                                       0)

# This is the same principle for Goal contributions per 90 minute matches (Goals + Assists)
if 'G+A' in df_clean.columns and '90s' in df_clean.columns:
    df_clean['GA_per_90'] = np.where(df_clean['90s'] > 0, 
                                      df_clean['G+A'] / df_clean['90s'], 
                                      0)

print(" Created per-90 statistics")

# ============================================


print("\n" + "="*60)
print("CLEANING SUMMARY")
print("="*60)

print(f"\n Final dataset: {len(df_clean)} player-season records") # how many rows in the cleaned dataframe
print(f" Columns: {len(df_clean.columns)}") # how many columns in the cleaned dataframe

# Show breakdown by league
print("\n^ Records by league:")
league_counts = df_clean['league'].value_counts().sort_index() # counts how many times each league appears in the 'league' column and sorts them alphabetically
for league, count in league_counts.items():
    print(f"   {league}: {count} player-seasons")

# Show breakdown by season
print("\n Records by season:")
season_counts = df_clean['season'].value_counts().sort_index() # same principle for seasons
for season, count in season_counts.items():
    print(f"   {season}: {count} player-seasons")

# This just shows a sample of the cleaned data with selected columns
print("\n Sample of cleaned data:")
sample_cols = ['Player', 'Pos', 'Squad', 'Age', 'MP', 'Gls', 'Ast', 'season', 'league']
sample_cols_exist = [col for col in sample_cols if col in df_clean.columns]
print(df_clean[sample_cols_exist].head(10))


# Show column list
print("\n Final columns:")
print(df_clean.columns.tolist()) # tolist() converts the Index object returned by df_clean.columns into a regular Python list

# ============================================
# SAVE CLEANED DATA
# ============================================

output_file = 'data/processed/fbref_cleaned.csv' # All this cleaned data is saved to a new CSV file called fbref_cleaned.csv
df_clean.to_csv(output_file, index=False) # to_csv() saves the dataframe to a CSV file 
# index=False means we don't want to save the row indices to the CSV file

print(f"\n Cleaned data saved to: {output_file}")
print(f"File size: {len(df_clean)} records × {len(df_clean.columns)} columns")

print("\n" + "="*60)
print(" CLEANING COMPLETE!")
print("="*60)

