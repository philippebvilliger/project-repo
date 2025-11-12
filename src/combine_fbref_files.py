import pandas as pd
import os
import glob

# ============================================
# COMBINE ALL FBREF CSV FILES
# ============================================

print("="*60)
print("LOADING FBREF DATA FROM CSV FILES")
print("="*60)

# Find all CSV files in data/raw/ folder
csv_files = glob.glob('data/raw/*.csv')

# Filter to only FBref files (exclude any other CSVs)
# FBref files should follow pattern: league_name_YYYY-YYYY.csv
fbref_files = []
for file_path in csv_files:
    filename = os.path.basename(file_path)
    # Check if it matches FBref naming pattern
    if any(league in filename.lower() for league in ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']):
        fbref_files.append(file_path)

print(f"\n📊 Found {len(fbref_files)} FBref CSV files")

if len(fbref_files) == 0:
    print("\n⚠️  No FBref CSV files found in data/raw/")
    print("Expected files like: premier_league_2023-2024.csv")
    exit(1)

# Dictionary to standardize league names
league_mapping = {
    'premier_league': 'Premier-League',
    'la_liga': 'La-Liga',
    'serie_a': 'Serie-A',
    'bundesliga': 'Bundesliga',
    'ligue_1': 'Ligue-1'
}

# List to store all dataframes
all_stats = []

# Load each CSV file
for file_path in sorted(fbref_files):
    filename = os.path.basename(file_path)
    filename_no_ext = filename.replace('.csv', '')
    
    # Extract league and season from filename
    # Example: "premier_league_2023-2024" → ["premier_league", "2023-2024"]
    parts = filename_no_ext.rsplit('_', 1)
    
    if len(parts) != 2:
        print(f"⚠️  Skipping {filename} - unexpected format")
        continue
    
    league_raw, season = parts
    league = league_mapping.get(league_raw.lower(), league_raw)
    
    print(f"Loading {league} {season}...", end=" ")
    
    try:
        # Load CSV with semicolon separator
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        
        # Check if empty
        if len(df) == 0:
            print(f"⚠️  Empty file!")
            continue
        
        # Remove duplicate header rows (where Player column = 'Player')
        if 'Player' in df.columns:
            df = df[df['Player'] != 'Player']
        
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
        
        # Add metadata columns
        df['season'] = str(season)
        df['league'] = str(league)
        
        # Append to list
        all_stats.append(df)
        
        print(f"✓ ({len(df)} players)")
        
    except Exception as e:
        print(f"✗ Error: {e}")

# ============================================
# COMBINE ALL DATA
# ============================================

if all_stats:
    print("\n" + "="*60)
    print("COMBINING ALL DATA")
    print("="*60)
    
    # Combine all dataframes
    fbref_stats = pd.concat(all_stats, ignore_index=True)
    
    print(f"\n✅ Total player-season records: {len(fbref_stats)}")
    
    # Data quality check
    print("\n🔍 Data Quality Check:")
    print(f"   Season data type: {fbref_stats['season'].dtype}")
    print(f"   League data type: {fbref_stats['league'].dtype}")
    
    # Show unique values
    unique_seasons = sorted(fbref_stats['season'].unique())
    unique_leagues = sorted(fbref_stats['league'].unique())
    
    print(f"\n   Unique seasons: {unique_seasons}")
    print(f"   Number of seasons: {len(unique_seasons)}")
    print(f"\n   Unique leagues: {unique_leagues}")
    print(f"   Number of leagues: {len(unique_leagues)}")
    
    # Show sample
    print("\n📋 Sample of combined data:")
    sample_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'MP', 'Gls', 'Ast', 'season', 'league']
    sample_cols_exist = [col for col in sample_cols if col in fbref_stats.columns]
    
    if sample_cols_exist:
        print(fbref_stats[sample_cols_exist].head(10))
    else:
        print(fbref_stats.head(10))
    
    print("\n📊 All columns:")
    print(fbref_stats.columns.tolist())
    
    # Save combined data
    output_file = 'data/raw/fbref_stats_raw.csv'
    fbref_stats.to_csv(output_file, index=False)
    
    print(f"\n✅ Combined data saved to: {output_file}")
    print(f"   Total records: {len(fbref_stats)}")
    
    # Show breakdown by league
    print("\n📈 Records by league:")
    league_counts = fbref_stats['league'].value_counts().sort_index()
    for league, count in league_counts.items():
        print(f"   {league}: {count} player-seasons")
    
    # Show breakdown by season
    print("\n📈 Records by season:")
    season_counts = fbref_stats['season'].value_counts().sort_index()
    for season, count in season_counts.items():
        print(f"   {season}: {count} player-seasons")
    
else:
    print("\n❌ No data loaded!")
    print("   Check that:")
    print("   1. CSV files are in data/raw/ folder")
    print("   2. Files are named correctly (e.g., premier_league_2023-2024.csv)")
    print("   3. Files contain valid FBref data")

print("\n" + "="*60)
print("✅ FBREF DATA LOADING COMPLETE!")
print("="*60)


