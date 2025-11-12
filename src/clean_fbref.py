import pandas as pd
import numpy as np

# ============================================
# CLEAN FBREF DATA
# ============================================

print("="*60)
print("CLEANING FBREF DATA")
print("="*60)

# Load the raw combined FBref data
input_file = 'data/processed/fbref_stats_raw.csv'
print(f"\n📂 Loading data from: {input_file}")

try:
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} records")
except FileNotFoundError:
    print(f"❌ Error: File not found at {input_file}")
    print("   Make sure you ran combine_fbref_files.py first!")
    exit(1)

print(f"\n📊 Original columns: {len(df.columns)}")
print(f"   Sample columns: {df.columns.tolist()[:10]}")

# ============================================
# STEP 1: Remove Unnamed columns
# ============================================

print("\n" + "-"*60)
print("STEP 1: Removing Unnamed columns")
print("-"*60)

unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
print(f"   Found {len(unnamed_cols)} Unnamed columns")

df = df.drop(columns=unnamed_cols)
print(f"✅ Removed Unnamed columns. Now have {len(df.columns)} columns")

# ============================================
# STEP 2: Keep only essential columns
# ============================================

print("\n" + "-"*60)
print("STEP 2: Selecting essential columns")
print("-"*60)

# Define the columns we want to keep
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
    'npxG',        # Non-penalty xG
    'xAG',         # Expected Assisted Goals
    'npxG+xAG',    # Non-penalty xG + xAG
    'PrgC',        # Progressive Carries
    'PrgP',        # Progressive Passes
    'PrgR',        # Progressive Receptions
    'season',      # Season
    'league'       # League
]

# Check which columns actually exist in our dataframe
available_cols = []
for col in essential_cols:
    # For columns that might have duplicates (Gls, Gls.1, Gls.2)
    # We take the first one
    matching_cols = [c for c in df.columns if c == col or c.startswith(col + '.')]
    if matching_cols:
        available_cols.append(matching_cols[0])  # Take first match
    elif col in df.columns:
        available_cols.append(col)

print(f"   Found {len(available_cols)} out of {len(essential_cols)} essential columns")
print(f"   Columns to keep: {available_cols}")

# Select only these columns
df_clean = df[available_cols].copy()

# Rename any columns with .1, .2 suffixes back to original names
rename_dict = {}
for col in df_clean.columns:
    if '.' in col and col.split('.')[0] in essential_cols:
        original_name = col.split('.')[0]
        rename_dict[col] = original_name

if rename_dict:
    print(f"\n   Renaming columns: {rename_dict}")
    df_clean = df_clean.rename(columns=rename_dict)

print(f"✅ Selected essential columns. Now have {len(df_clean.columns)} columns")

# ============================================
# STEP 3: Clean data types and values
# ============================================

print("\n" + "-"*60)
print("STEP 3: Cleaning data types and values")
print("-"*60)

# Remove rows where Player is null or empty
before_count = len(df_clean)
df_clean = df_clean[df_clean['Player'].notna()]
df_clean = df_clean[df_clean['Player'] != '']
after_count = len(df_clean)
if before_count != after_count:
    print(f"   Removed {before_count - after_count} rows with missing player names")

# Clean numeric columns
numeric_cols = ['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A', 
                'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR']

# Add xG columns if they exist
if 'xG' in df_clean.columns:
    numeric_cols.extend(['xG', 'npxG', 'xAG', 'npxG+xAG'])
if 'PrgC' in df_clean.columns:
    numeric_cols.extend(['PrgC', 'PrgP', 'PrgR'])

# Only process columns that exist
numeric_cols = [col for col in numeric_cols if col in df_clean.columns]

print(f"   Converting {len(numeric_cols)} columns to numeric")

for col in numeric_cols:
    # Replace commas with nothing (for numbers like "1,234")
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].astype(str).str.replace(',', '').str.replace(' ', '')
    
    # Convert to numeric, replacing errors with NaN
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Fill NaN with 0 for stats (but not Age)
    if col != 'Age':
        df_clean[col] = df_clean[col].fillna(0)

# Clean Age specifically - if missing, we might drop these rows
df_clean = df_clean[df_clean['Age'].notna()]

# Convert Age to integer
df_clean['Age'] = df_clean['Age'].astype(int)

print("✅ Converted numeric columns and cleaned values")

# ============================================
# STEP 4: Standardize player names
# ============================================

print("\n" + "-"*60)
print("STEP 4: Standardizing player names")
print("-"*60)

# Remove leading/trailing whitespace
df_clean['Player'] = df_clean['Player'].str.strip()

# Remove any special characters that might cause matching issues
# But keep accents (é, ñ, etc.) as they're part of names
df_clean['Player'] = df_clean['Player'].str.replace(r'\s+', ' ', regex=True)

print("✅ Standardized player names")

# ============================================
# STEP 5: Remove duplicate rows
# ============================================

print("\n" + "-"*60)
print("STEP 5: Removing duplicates")
print("-"*60)

before_dedup = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['Player', 'Squad', 'season', 'league'], keep='first')
after_dedup = len(df_clean)

if before_dedup != after_dedup:
    print(f"   Removed {before_dedup - after_dedup} duplicate records")
else:
    print("   No duplicates found")

# ============================================
# STEP 6: Create useful derived columns
# ============================================

print("\n" + "-"*60)
print("STEP 6: Creating derived columns")
print("-"*60)

# Goals per 90
if 'Gls' in df_clean.columns and '90s' in df_clean.columns:
    df_clean['Gls_per_90'] = np.where(df_clean['90s'] > 0, 
                                       df_clean['Gls'] / df_clean['90s'], 
                                       0)

# Assists per 90
if 'Ast' in df_clean.columns and '90s' in df_clean.columns:
    df_clean['Ast_per_90'] = np.where(df_clean['90s'] > 0, 
                                       df_clean['Ast'] / df_clean['90s'], 
                                       0)

# Goal contributions per 90 (Goals + Assists)
if 'G+A' in df_clean.columns and '90s' in df_clean.columns:
    df_clean['GA_per_90'] = np.where(df_clean['90s'] > 0, 
                                      df_clean['G+A'] / df_clean['90s'], 
                                      0)

print("✅ Created per-90 statistics")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*60)
print("CLEANING SUMMARY")
print("="*60)

print(f"\n✅ Final dataset: {len(df_clean)} player-season records")
print(f"✅ Columns: {len(df_clean.columns)}")

# Show breakdown by league
print("\n📈 Records by league:")
league_counts = df_clean['league'].value_counts().sort_index()
for league, count in league_counts.items():
    print(f"   {league}: {count} player-seasons")

# Show breakdown by season
print("\n📈 Records by season:")
season_counts = df_clean['season'].value_counts().sort_index()
for season, count in season_counts.items():
    print(f"   {season}: {count} player-seasons")

# Show sample
print("\n📋 Sample of cleaned data:")
sample_cols = ['Player', 'Pos', 'Squad', 'Age', 'MP', 'Gls', 'Ast', 'season', 'league']
sample_cols_exist = [col for col in sample_cols if col in df_clean.columns]
print(df_clean[sample_cols_exist].head(10))

# Show column list
print("\n📊 Final columns:")
print(df_clean.columns.tolist())

# ============================================
# SAVE CLEANED DATA
# ============================================

output_file = 'data/processed/fbref_cleaned.csv'
df_clean.to_csv(output_file, index=False)

print(f"\n✅ Cleaned data saved to: {output_file}")
print(f"   File size: {len(df_clean)} records × {len(df_clean.columns)} columns")

print("\n" + "="*60)
print("✅ CLEANING COMPLETE!")
print("="*60)

