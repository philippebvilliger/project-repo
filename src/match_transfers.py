import pandas as pd # we import pandas library for data manipulation
import numpy as np # we import numpy for fumerical operations
from fuzzywuzzy import fuzz # This imports fuzzy string matching functions to calculate similarity between strings.
from fuzzywuzzy import process # This imports the higher-level functions that find the best match from a list of strings
from glob import glob # This is a tool to find files matching a certain path

# ============================================
print("="*60)
print("MATCHING TRANSFERMARKT & FBREF DATA")
print("TWO-PASS STRATEGY: BEFORE + AFTER")
print("="*60)

# We start off by loading and reding the filtered combined transfermarkt file
print("\n📂 Loading Transfermarkt data...")
transfers = pd.read_csv('data/transfers_filtered.csv')
print(f"✅ Loaded {len(transfers)} transfers")

# ============================================
print("\n📂 Loading ALL FBref data...")

fbref_files = glob('data/fbref/*.csv') # This finds all our fbref files that are in the fbref folder i.e., for each season and league
fbref_data = [] # we create an empty list to store dataframes from each Fbref csv file

for file in fbref_files: # we loop over each fbref file
    try: # We create a try-except statement in case we can't read the file
        df = pd.read_csv(file)
        
        # This allows us to extract just the file name with the preceding path and we also remove the '.csv' extension
        filename = file.split('/')[-1].replace('.csv', '')
        
        if 'premier' in filename.lower(): # so for each file it check which league it belongs to based on what's written in the name and add the name of the league to each row i.e., for each player record
            df['fbref_league'] = 'Premier League'
        elif 'laliga' in filename.lower():
            df['fbref_league'] = 'La Liga'
        elif 'serie' in filename.lower():
            df['fbref_league'] = 'Serie A'
        elif 'bundesliga' in filename.lower():
            df['fbref_league'] = 'Bundesliga'
        elif 'ligue' in filename.lower():
            df['fbref_league'] = 'Ligue 1'
        
        # Extract season (e.g., "2022-2023" → start year 2022)
        season_parts = filename.split('_')[-1].split('-')
        if len(season_parts) == 2: # this ensures a valid season format as there should only be 2 elements in this list after separation
            df['fbref_season'] = int(season_parts[0]) # stores the starting year of the season in a new column
        
        fbref_data.append(df) # Once all of that is done we can add it to our combined dataframe list
        print(f"   ✓ Loaded {file}")
    except Exception as e: # in case you can't load it.
        print(f"   ✗ Failed to load {file}: {e}")

fbref = pd.concat(fbref_data, ignore_index=True) # Once you have looped over every single dataframe and added it to the combined df list, you can join them together via concat()
print(f"✅ Total FBref records: {len(fbref)}")

# ============================================

print("\n🔧 Preprocessing data...")

# We now have our combined files with all information relevant to transfers from transfermarkt and all information relevant to football stats from fbref

# We now have to prepare the player's names for the fuzzy matching as we will match them between these 2 different combined files
transfers['player_clean'] = transfers['Player'].str.lower().str.strip() # all these functions should make them as similar as possible to facilitate the matching process
fbref['player_clean'] = fbref['Player'].str.lower().str.strip()

# Extract transfer year
transfers['transfer_year'] = transfers['source_file'].str.extract(r'(\d{4})').astype(int) # The regex here allows use to extract the 4-digit year from the source_file column. 
# This is of particular importance because we want to be able to match the player for the correct season i.e., before and after the transfer
# The source_file column contains the filename from which the transfer record came, which usually has the season or year in it. e.g., "premier_league_2022-2023.csv"
# We want to store this as an integer because  we will subtract 1 to get the before season. 

# Calculate before and after seasons
transfers['season_before'] = transfers['transfer_year'] - 1
transfers['season_after'] = transfers['transfer_year']

# We define a standardize function in order to obtain the same homogeneous names for each league in order to avoid confusion
def standardize_league(league):
    league = str(league)
    if 'Liga' in league or 'La-Liga' in league or 'La Liga' in league:
        return 'La Liga'
    elif 'Premier' in league:
        return 'Premier League'
    elif 'Serie' in league:
        return 'Serie A'
    elif 'Bundesliga' in league:
        return 'Bundesliga'
    elif 'Ligue' in league:
        return 'Ligue 1'
    return league

transfers['league_clean'] = transfers['league'].apply(standardize_league) # this simply applies this newly created function to the values of 'league' in the combined transfer files as they weren't initially the same as in fbref
#apply() takes a function and applies it to each value in the column

print("✅ Preprocessing complete")

# This allows us to remove rows player_clean has a missing value i.e., NaN
transfers = transfers[transfers['player_clean'].notna()] # notna() returns false if there is a missing value and will hence be omitted from the updated combined file
fbref = fbref[fbref['player_clean'].notna()] # same for fbref

print(f"✅ After cleaning: {len(transfers)} transfers, {len(fbref)} FBref records")

# ============================================
print("\n🔗 PASS 1: Matching BEFORE season stats...")

def find_best_match(name, choices, threshold=78): # we define this function where name is the player you want to march from transfermarkt and choices is the list of names to match against from fbref
    # threshold = 78 means that a match will occur only if it's 78% similar we reason as such because same names can be written differently e.g., C. Ronaldo and Cristiano Ronaldo
    """Find best matching name using fuzzy string matching"""
    # Handle missing/invalid names
    if pd.isna(name) or not isinstance(name, str) or len(choices) == 0: # pd.isna returns true if the name is missing
        return None, 0
        # So here we return no matches if the name is missing or if name is not a string or if the list from fbref is empty meaning no list of names to match against from fbref
    
    # Filter out any NaN values from choices
    valid_choices = [c for c in choices if pd.notna(c) and isinstance(c, str)] # This creates a cleaned list of choices to compare against. Only includes choices that are not misssing and are strings
    
    if len(valid_choices) == 0: # if the list of valid choices is empty we return no best matches obviously
        return None, 0
    
    result = process.extractOne(name, valid_choices, scorer=fuzz.ratio) # This is a fuzzywuzzy function that allows us to find the closest match
    # This function extractOne concretely compares name to each string in our list of valid_choices
    # scorer = fuzz.ratio means that we use the Levenshtein distance ratio to measure similarity i.e., a scale from 0 to 100
    # This returns a tuple  with the best match out of the valid_choices list and its respective similarity score
    if result and result[1] >= threshold: # if we found a match and its similarity score exceeds the threshold of 78 then, we return the match
        return result[0], result[1]
    return None, 0 # otherwise we return None

before_matches = {} # We will first be focusing on matching before season starts i.e., we want the stats of the player before the transfer occured in order to compare the before and after

for idx, transfer in transfers.iterrows(): # .iterrows() is a method that allows you to loop over each row in the dataframe
    player_name = transfer['player_clean']
    dest_league = transfer['league_clean']
    season_before = transfer['season_before']
    # You extract the cleaned player name, destination league, and season before transfer
    
    # Now for the fbref data, you are making a boolean series in order to match the league and season of transfermarkt
    # if the league of the fbref is the dest_league of transfermarkt df then return True. Same goes for season.
    # This will make a new filtered dataframe as now pandas will only return the rows where both boolean are true i.e., where both match
    fbref_filtered = fbref[
        (fbref['fbref_league'] == dest_league) & 
        (fbref['fbref_season'] == season_before)
    ]
    
    if len(fbref_filtered) > 0: # if the newly filtered df is not empty
        fbref_names = fbref_filtered['player_clean'].unique() # It extracts the player names from the filtered df and via unique() ensures that we only get each name once in order to avoid duplicates
        # We now have a valid player name that we will try to match against
        best_match, score = find_best_match(player_name, fbref_names) 
        # we now use the previously created function with the player name from transfermarkt to match against the candidate fbref_names i.e., a list of player names that have been filtered to match the season and league of player's transfer season
        
        if best_match: # if there actually is a match based on previously defined criteria
            fbref_stats = fbref_filtered[fbref_filtered['player_clean'] == best_match].iloc[0] # we select the row where the player’s cleaned name equals best_match. 
            # .iloc[0] → selects the first row from the result (in case multiple rows match).
            before_matches[idx] = fbref_stats.to_dict()
            # This converts the fbref_stats Series into a dictionary using .to_dict().
        # If best_match is None, it means no sufficiently similar name was found so we skip this player
    
    if (idx + 1) % 50 == 0:  # Prints a progress message every 50 rows
        print(f"   Processed {idx + 1}/{len(transfers)} transfers...")

print(f"✅ BEFORE stats matched: {len(before_matches)}/{len(transfers)} ({len(before_matches)/len(transfers)*100:.1f}%)")

# ============================================
print("\n🔗 PASS 2: Matching AFTER season stats...")

after_matches = {} # We now create a dictionnary for the after transfer stats as we want to see the shift in performance following the transfer.

for idx, transfer in transfers.iterrows():
    player_name = transfer['player_clean']
    dest_league = transfer['league_clean']
    season_after = transfer['season_after']
    # Same process as before you extract the cleaned player name, destination league, but this time the season after the transfer
    
    # Same as before, this makes a new filtered dataframe as now pandas will only return the rows where both boolean are true i.e., where both match
    fbref_filtered = fbref[
        (fbref['fbref_league'] == dest_league) & 
        (fbref['fbref_season'] == season_after)
    ]
    
    if len(fbref_filtered) > 0:
        fbref_names = fbref_filtered['player_clean'].unique()
        best_match, score = find_best_match(player_name, fbref_names)
        # # we now use the previously created function with the player name from transfermarkt to match against the candidate fbref_names i.e., a list of player names that have been filtered to match the season and league of player's transfer season
        
        if best_match:
            fbref_stats = fbref_filtered[fbref_filtered['player_clean'] == best_match].iloc[0] # we select the row where the player’s cleaned name equals best_match. 
            # .iloc[0] → selects the first row from the result (in case multiple rows match).
            after_matches[idx] = fbref_stats.to_dict() # This converts the fbref_stats Series into a dictionary using .to_dict().
        # If best_match is None, it means no sufficiently similar name was found so we skip this player
    
    if (idx + 1) % 50 == 0:
        print(f"   Processed {idx + 1}/{len(transfers)} transfers...") # Prints a progress message every 50 rows

print(f"✅ AFTER stats matched: {len(after_matches)}/{len(transfers)} ({len(after_matches)/len(transfers)*100:.1f}%)")

# ============================================
print("\n📊 Creating final datasets...")

# Dataset 1: Players with BOTH before AND after stats (complete comparison)
complete_matches = [] # This a list that will store dictionnaries containing Transfermarkt data for the player, FBref stats before the transfer and FBref stats after the transfer

for idx in before_matches.keys(): # this loops over all indices that have a "before transfer" match
    if idx in after_matches: # It checks if the same transfer also has a valid “after transfer” match in FBref if so:
        transfer = transfers.loc[idx]  # Gives us the transfermarkt row corresponding to this index
        # .loc[idx] selects the row by index label and returns a Pandas Series containing the player’s transfer data
        before_stats = before_matches[idx] # Gets the before transfer fbref stats dictionnary for this player
        after_stats = after_matches[idx] # Gets the after transfer fbref stats dictionnary for this player
        
        complete_record = { # If there is a before and after transfer match for the same player then, it creates a single dictionary combining all data for this player
            **transfer.to_dict(), # converts the transfermarkt row corresponding to the index we are looping on into a dictionnary
            **{f'before_{k}': v for k, v in before_stats.items()}, # We make a new dicitonnary where each key is prefixed with before so we are able to separate the before and after transfer stats
            **{f'after_{k}': v for k, v in after_stats.items()} # Here for after transfer stats
            # Given the fact that we are creating a dictionnary out of 3 diffrent ones, we use ** to able to merge all 3 of them into one single dictionnary
        }
        complete_matches.append(complete_record) # Now each player who had a before match and an after match now has a dictionnary with all his stats being transfer related as well as before and after season performance
        # We now add each of these dictionnaries to the list

print(f"✅ Complete matches (BEFORE + AFTER): {len(complete_matches)}")

# Dataset 2: All matches (with whatever data available)
all_matches = []
for idx in transfers.index:  # Changed to iterate over index instead of iterrows
    transfer = transfers.loc[idx]  # Changed from .iloc to .loc
    record = transfer.to_dict()
    record['has_before'] = idx in before_matches
    record['has_after'] = idx in after_matches
    
    if idx in before_matches:
        record.update({f'before_{k}': v for k, v in before_matches[idx].items()})
    if idx in after_matches:
        record.update({f'after_{k}': v for k, v in after_matches[idx].items()})
    
    all_matches.append(record)

# ============================================
print("\n📈 Final statistics:")
print(f"   Total transfers: {len(transfers)}")
print(f"   With BEFORE stats: {len(before_matches)} ({len(before_matches)/len(transfers)*100:.1f}%)")
print(f"   With AFTER stats: {len(after_matches)} ({len(after_matches)/len(transfers)*100:.1f}%)")
print(f"   With BOTH (complete): {len(complete_matches)} ({len(complete_matches)/len(transfers)*100:.1f}%)")

print("\n📊 Complete matches by league:")
complete_df = pd.DataFrame(complete_matches)
if len(complete_df) > 0:
    for league in complete_df['league_clean'].unique():
        league_count = len(complete_df[complete_df['league_clean'] == league])
        league_total = len(transfers[transfers['league_clean'] == league])
        print(f"   {league}: {league_count}/{league_total} ({league_count/league_total*100:.1f}%)")

# ============================================
print("\n💾 Saving results...")

# Save complete matches (BEFORE + AFTER)
complete_df = pd.DataFrame(complete_matches)
complete_df.to_csv('data/processed/transfers_matched_complete.csv', index=False)
print(f"✅ Complete matches saved: data/processed/transfers_matched_complete.csv")

# Save all matches
all_df = pd.DataFrame(all_matches)
all_df.to_csv('data/processed/transfers_matched_all.csv', index=False)
print(f"✅ All matches saved: data/processed/transfers_matched_all.csv")

# Save summary
unmatched = transfers[~transfers.index.isin(before_matches.keys()) & ~transfers.index.isin(after_matches.keys())]
unmatched.to_csv('data/processed/transfers_unmatched.csv', index=False)
print(f"✅ Unmatched transfers saved: data/processed/transfers_unmatched.csv")

print("\n" + "="*60)
print("✅ MATCHING COMPLETE!")
print("="*60)
print(f"\nYour analysis-ready dataset: {len(complete_matches)} players with complete BEFORE/AFTER comparison")
