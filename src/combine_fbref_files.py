import pandas as pd # pandas library for data manipulation
import os  # os library for file path operations such as joining paths
import glob # glob library for file pattern matching e.g., finding all CSV files in a directory

# ============================================


print("="*60)
print("LOADING FBREF DATA FROM CSV FILES")
print("="*60)

# The glob() function is used to find all CSV files in the specified directory 
# So here we find all csv files in the fbref folder corresponding to different leagues and seasons e.g., premier_league_2023-2024.csv
csv_files = glob.glob('data/fbref/*.csv')


fbref_files = [] # we make a list to store the paths of the fbref files
for file_path in csv_files: # iterate through all found csv files
    filename = os.path.basename(file_path) # extract the filename from the full path via basename() that removes the directory path
    # Check if it matches FBref naming pattern
    if any(league in filename.lower() for league in ['bundesliga', 'laliga', 'ligue1', 'premier_league', 'seriea']):
        fbref_files.append(file_path)
        # any() returns True if any element of the iterable is true. 
        # Here we check if any of the league names are in the filename 
        # if it matches any of the league names, we add it to the fbref_files list

print(f"\n Found {len(fbref_files)} FBref CSV files")

if len(fbref_files) == 0:
    print("\n  No FBref CSV files found in data/fbref/")
    print("Expected files like: premier_league_2023-2024.csv")
    exit(1) # exit(1) indicates an error occurred so only if there are no fbref files found

# We make a dictionary to map league codes to full league names as they have different naming conventions in filenames e.g., 'premier_league' to 'Premier-League'
league_mapping = {
    'bundesliga': 'Bundesliga',
    'laliga': 'La-Liga',
    'ligue1': 'Ligue-1',
    'premier_league': 'Premier-League',
    'seriea': 'Serie-A'
}

# This list we will hold all the dataframes we load from each CSV file
all_stats = []

# Load each CSV file
for file_path in sorted(fbref_files): # sorted() to process files in order
    filename = os.path.basename(file_path)
    filename_no_ext = filename.replace('.csv', '') # we remove the .csv extension to extract league and season info
    
    # Extract league and season from filename
    # E.g., "premier_league_2023-2024" → ["premier_league", "2023-2024"]
    parts = filename_no_ext.rsplit('_', 1) # .rsplit() splits a string into a list so we divide the name by the league and season 
    
    if len(parts) != 2: # if we end up with an unexpected format we skip it
        print(f"  Skipping {filename} - unexpected format")
        continue # continue skips to the next loop iteration
    
    league_raw, season = parts  # league_raw is the name of the league in the given file
    league = league_mapping.get(league_raw.lower(), league_raw) # get() retrieves the value for a given key, here the key correspond the basename of the file
    # in return it gives us the value i.e., the basename written correctly

    print(f"Loading {league} {season}...", end=" ")
    
    try: # we use a try-except statement in case there is a problem reading the csv file
        # We load the csv file into a pandas dataframe
        df = pd.read_csv(file_path, sep=',', encoding='utf-8') # encoding will be utf-8 and the separator will be commas
        
        # If the the dataframe has no rows we skip it
        if len(df) == 0:
            print(f"  Empty file!")
            continue
        
        if 'Player' in df.columns:
            df = df[df['Player'] != 'Player'] # We remove duplicate header rows (where Player column = 'Player') by keeping only the different ones
        
        # Some csv files have multiple level headers e.g., a header and then underneath a more detailed under header.
        if isinstance(df.columns, pd.MultiIndex): # we check whether the columns of this dataframe belongs to the class multiIndex from pandas library
            df.columns = ['_'.join(col).strip('_') for col in df.columns.values] # if so, we get the values of the column index. For a multiindex that would be a list of tuples
            # We loop over each tuple in that list and we join them by an underscore. This gives us the names of the dataframe's columns
        
        # We add the same value to each of these columns for a given dataframe as they remain the same for all players obviously
        df['season'] = str(season)
        df['league'] = str(league)
        
        # Append to list
        all_stats.append(df) # once it's all done, we can add our newly adjusted dataframe to the global list
        
        print(f"✓ ({len(df)} players)")
        
    except Exception as e: # in case we can't load the csv file
        print(f"✗ Error: {e}")

# ============================================

# Once we have loaded and adjusted every csv file we can check whether it has been added to our list and if so, we can combine all the data.
if all_stats: 
    print("\n" + "="*60)
    print("COMBINING ALL DATA")
    print("="*60)
    
    # Combine all dataframes
    fbref_stats = pd.concat(all_stats, ignore_index=True) # this pandas function joins together all dataframes into a single one
    
    print(f"\n Total player-season records: {len(fbref_stats)}")
    
    # Data quality check
    print("\n Data Quality Check:")
    print(f"   Season data type: {fbref_stats['season'].dtype}") # this gives us the data type for each season and league in order to insure that they were loaded as strings not to be use as numeric data
    print(f"   League data type: {fbref_stats['league'].dtype}")
    
    # Show unique values
    unique_seasons = sorted(fbref_stats['season'].unique()) # This basically gives us a summary of all the different leagues and seasons in the dataframe
    unique_leagues = sorted(fbref_stats['league'].unique())
    
    print(f"\n   Unique seasons: {unique_seasons}")
    print(f"   Number of seasons: {len(unique_seasons)}")
    print(f"\n   Unique leagues: {unique_leagues}")
    print(f"   Number of leagues: {len(unique_leagues)}")
    
    # We want to make a sample but before we want to ascertain whether these sample columns are in our combined dataframe
    print("\n Sample of combined data:")
    sample_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'MP', 'Gls', 'Ast', 'season', 'league']
    sample_cols_exist = [col for col in sample_cols if col in fbref_stats.columns] 
    
    if sample_cols_exist: # For the columns that do exist we get a sample of the first 10 rows
        print(fbref_stats[sample_cols_exist].head(10))
    else:
        print(fbref_stats.head(10))
    
    print("\n All columns:")
    print(fbref_stats.columns.tolist()) # this gives us a list of all the columns in our newly combined dataframe
    
    # This creares a processed folder in our repo if it doesn't already exist
    os.makedirs('data/processed', exist_ok=True)
    
    # We save this combined dataframe in this newly created processed folder and call it fbref_stats_raw.csv
    output_file = 'data/processed/fbref_stats_raw.csv'
    fbref_stats.to_csv(output_file, index=False)
    
    print(f"\n Combined data saved to: {output_file}")
    print(f"   Total records: {len(fbref_stats)}")
    
    # Show breakdown by league
    print("\n Records by league:")
    league_counts = fbref_stats['league'].value_counts().sort_index() # this line counts how many rows i.e., player records belong to each league
    for league, count in league_counts.items(): # .items() lets you iterate over the (key, value) pairs of a Series:
        print(f"   {league}: {count} player-seasons")
    
    # Show breakdown by season
    print("\n Records by season:")
    season_counts = fbref_stats['season'].value_counts().sort_index() # the same principle for each season here
    for season, count in season_counts.items():
        print(f"   {season}: {count} player-seasons")
    
else: # if there are no dataframes in the list all_stats
    print("\n No data loaded!")
    print("   Check that:")
    print("   1. CSV files are in data/fbref/ folder")
    print("   2. Files are named correctly (e.g., premier_league_2023-2024.csv)")
    print("   3. Files contain valid FBref data")

print("\n" + "="*60)
print(" FBREF DATA LOADING COMPLETE!")
print("="*60)
