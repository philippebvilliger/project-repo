import pandas as pd # We import pandas for data manipulation
import numpy as np # We import numpy for numerical operations such as handling NaN values.

print("="*60)
print("CLEANING FBREF PERFORMANCE DATA")
print("="*60)

# ============================================
# STEP 1: LOAD RAW FBREF DATA
# ============================================

print("\n📊 Loading raw FBref data...")
df_raw = pd.read_csv('data/raw/fbref_stats_raw.csv')

print(f"✅ Loaded {len(df_raw)} team-season records")
print(f"   Columns: {len(df_raw.columns)}")

# ============================================
# STEP 2: FIX COLUMN NAMES
# ============================================

print("\n🔧 Fixing column names...")

# The raw data has multi-level headers that need to be flattened
# We'll rename the key columns we need for analysis

# First, let's see what columns we have
print("\nOriginal columns (first 20):")
for i, col in enumerate(df_raw.columns[:20]):
    print(f"  {i}: {col}")

# Create a mapping of old column names to new clean names
# The columns are positional, so we'll use their index positions
column_mapping = {
    df_raw.columns[0]: 'Team',           # Squad name
    df_raw.columns[1]: 'Players_Used',   # Number of players
    df_raw.columns[2]: 'Avg_Age',        # Average age
    df_raw.columns[3]: 'Possession',     # Possession %
    df_raw.columns[4]: 'Matches',        # Matches played
    df_raw.columns[5]: 'Starts',         # Starts
    df_raw.columns[6]: 'Minutes',        # Minutes played
    df_raw.columns[7]: 'Minutes_90s',    # 90s played
    df_raw.columns[8]: 'Goals',          # Goals
    df_raw.columns[9]: 'Assists',        # Assists
    df_raw.columns[10]: 'Goals_Assists', # G+A
    df_raw.columns[11]: 'Goals_PK',      # Goals - penalties
    df_raw.columns[12]: 'PK',            # Penalties made
    df_raw.columns[13]: 'PKatt',         # Penalties attempted
    df_raw.columns[14]: 'Yellow',        # Yellow cards
    df_raw.columns[15]: 'Red',           # Red cards
    df_raw.columns[16]: 'Goals_per90',   # Goals per 90
    df_raw.columns[17]: 'Assists_per90', # Assists per 90
    df_raw.columns[18]: 'G+A_per90',     # G+A per 90
    df_raw.columns[19]: 'G-PK_per90',    # Non-penalty goals per 90
    df_raw.columns[20]: 'G+A-PK_per90',  # G+A-PK per 90
}

# Add season and league columns (these are at the end)
column_mapping[df_raw.columns[-2]] = 'Season'
column_mapping[df_raw.columns[-1]] = 'League'

# Find xG columns if they exist (they appear from 2017-2018 onwards)
for i, col in enumerate(df_raw.columns):
    col_str = str(col).lower()
    if 'xg' in col_str and 'xag' not in col_str and 'npxg' not in col_str:
        if i not in column_mapping:
            column_mapping[df_raw.columns[i]] = 'xG'
    elif 'npxg' in col_str and 'xag' not in col_str:
        if i not in column_mapping:
            column_mapping[df_raw.columns[i]] = 'npxG'
    elif 'xag' in col_str and 'npxg' not in col_str:
        if i not in column_mapping:
            column_mapping[df_raw.columns[i]] = 'xAG'
    elif 'npxg+xag' in col_str or 'npxg_xag' in col_str:
        if i not in column_mapping:
            column_mapping[df_raw.columns[i]] = 'npxG+xAG'

# Rename columns
df = df_raw.rename(columns=column_mapping)

print(f"\n✅ Renamed {len(column_mapping)} columns")

# ============================================
# STEP 3: SELECT RELEVANT COLUMNS
# ============================================

print("\n🎯 Selecting relevant columns for analysis...")

# Define the columns we want to keep
core_columns = ['Team', 'Season', 'League', 'Players_Used', 'Avg_Age', 
                'Possession', 'Matches', 'Minutes', 'Minutes_90s']

performance_columns = ['Goals', 'Assists', 'Goals_Assists', 'Goals_PK', 
                       'PK', 'PKatt', 'Yellow', 'Red']

per90_columns = ['Goals_per90', 'Assists_per90', 'G+A_per90', 
                 'G-PK_per90', 'G+A-PK_per90']

# Add xG columns if they exist
xg_columns = []
for col in ['xG', 'npxG', 'xAG', 'npxG+xAG']:
    if col in df.columns:
        xg_columns.append(col)

# Combine all columns we want to keep
columns_to_keep = core_columns + performance_columns + per90_columns + xg_columns

# Keep only columns that exist in our dataframe
columns_to_keep = [col for col in columns_to_keep if col in df.columns]

df_clean = df[columns_to_keep].copy()

print(f"✅ Selected {len(columns_to_keep)} columns")
print("\nColumns kept:")
for col in columns_to_keep:
    print(f"  - {col}")

# ============================================
# STEP 4: CLEAN AND STANDARDIZE DATA
# ============================================

print("\n🧹 Cleaning data...")

# Convert numeric columns to proper types
numeric_columns = [col for col in df_clean.columns if col not in ['Team', 'Season', 'League']]

for col in numeric_columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Handle missing values in xG columns (they don't exist for 2016-2017)
for col in xg_columns:
    if col in df_clean.columns:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            print(f"  ⚠️  {col}: {missing_count} missing values (likely 2016-2017 season)")

# Standardize team names for matching with transfer data
# Common variations we need to handle
team_name_mapping = {
    'Manchester Utd': 'Manchester United',
    'Manchester City': 'Manchester City',
    'Tottenham': 'Tottenham Hotspur',
    'Brighton': 'Brighton & Hove Albion',
    'Wolves': 'Wolverhampton Wanderers',
    "Nott'ham Forest": 'Nottingham Forest',
    'Newcastle Utd': 'Newcastle United',
    'West Ham': 'West Ham United',
    'Paris S-G': 'Paris Saint-Germain',
    'Eint Frankfurt': 'Eintracht Frankfurt',
    'Bayern Munich': 'Bayern München',
    'Gladbach': 'Borussia Mönchengladbach',
    'Dortmund': 'Borussia Dortmund',
    'Leverkusen': 'Bayer Leverkusen',
    'RB Leipzig': 'RB Leipzig',
}

df_clean['Team_Original'] = df_clean['Team']  # Keep original for reference
df_clean['Team'] = df_clean['Team'].replace(team_name_mapping)

print(f"\n✅ Standardized {len(team_name_mapping)} team names")

# ============================================
# STEP 5: ADD USEFUL DERIVED COLUMNS
# ============================================

print("\n➕ Adding derived columns...")

# Add a combined season-league identifier for easy filtering
df_clean['Season_League'] = df_clean['Season'] + '_' + df_clean['League']

# Calculate some useful ratios (only where we have the data)
if 'xG' in df_clean.columns and 'Goals' in df_clean.columns:
    df_clean['xG_Overperformance'] = df_clean['Goals'] - df_clean['xG']
    print("  ✓ Added xG_Overperformance")

if 'xAG' in df_clean.columns and 'Assists' in df_clean.columns:
    df_clean['xAG_Overperformance'] = df_clean['Assists'] - df_clean['xAG']
    print("  ✓ Added xAG_Overperformance")

# ============================================
# STEP 6: DATA QUALITY CHECKS
# ============================================

print("\n" + "="*60)
print("DATA QUALITY SUMMARY")
print("="*60)

print(f"\n✅ Total records: {len(df_clean)}")
print(f"✅ Seasons covered: {df_clean['Season'].nunique()}")
print(f"✅ Leagues: {df_clean['League'].nunique()}")
print(f"✅ Unique teams: {df_clean['Team'].nunique()}")

print("\n📊 Records by season:")
season_counts = df_clean['Season'].value_counts().sort_index()
for season, count in season_counts.items():
    print(f"  {season}: {count} teams")

print("\n📊 Records by league:")
league_counts = df_clean['League'].value_counts()
for league, count in league_counts.items():
    print(f"  {league}: {count} team-seasons")

# Check for missing data
print("\n⚠️  Missing data summary:")
missing_summary = df_clean.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
if len(missing_summary) > 0:
    for col, count in missing_summary.items():
        pct = (count / len(df_clean)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
else:
    print("  ✓ No missing data!")

# ============================================
# STEP 7: SAVE CLEANED DATA
# ============================================

print("\n" + "="*60)
print("SAVING CLEANED DATA")
print("="*60)

# Save cleaned data
output_file = 'data/processed/fbref_cleaned.csv'
df_clean.to_csv(output_file, index=False)

print(f"\n✅ Cleaned data saved to: {output_file}")
print(f"   Total records: {len(df_clean)}")
print(f"   Columns: {len(df_clean.columns)}")

# Display sample of cleaned data
print("\n📋 Sample of cleaned data (first 5 rows):")
print(df_clean.head())

print("\n" + "="*60)
print("✅ FBREF DATA CLEANING COMPLETE!")
print("="*60)

print("\n📌 Next steps:")
print("1. Load your transfer data (transfers_filtered.csv)")
print("2. Match players between transfer data and FBref data")
print("3. Extract before/after performance for each transfer")
