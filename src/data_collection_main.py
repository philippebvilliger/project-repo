import pandas as pd # we import pandas library for data manipulation
import os # we import os to allow us to work with file paths and directories
from glob import glob # this will allow us to find all files having a specific pattern i.e., csv files 

# ============================================

print("="*60)
print("TRANSFER VALUE ANALYSIS")
print("="*60)


# I have gone on transfermarkt.com and compiled the data for transfers in the top 5 European leagues
# for seasons 2017-2018 to 2023-2024. I filtered for transfers involving attackers and midfielders and transfer fees of €5M+.
# I then turned them into csv files for each respective league and season. I saved them in the data/transfermarkt/ folder.
csv_files = glob('data/transfermarkt/*.csv')  # glob as we said allows us to import all csv files in the specified folder.

print(f"\nFound {len(csv_files)} CSV files:")
for file in csv_files:   # Loop through each file and print its name so user can see what's being loaded
    print(f"   - {file}") 

# Load and combine all CSV files
all_transfers = []  # we will be storing all transfers in this list, in order to have one big dataset.
#This makes it easier to analyze and filter the data later on otherwise, we would have to analyze each csv file separately.

for file in csv_files: # We want to loop through each csv file found in the data/transfermarkt/ folder and load them one by one.
    try: # We use try in case there are any errors in loading a specific file.
        df = pd.read_csv(file) # We read the csv file and then turn it into a pandas dataframe for easier manipulation.
        
        # A dataframe df is like a table with rows and columns, similar to an Excel spreadsheet.

        # Remove any completely empty columns that might have been created by trailing commas
        df = df.dropna(axis=1, how='all')
        
        # Add source information
        filename = os.path.basename(file) # For each csv file, we extract its filename to keep track of where the data came from.
        df['source_file'] = filename # We add a new column to the dataframe called 'source_file' to store the filename.
        
        # We extract league and season from filename for context. Important!
        # Example: "premier_league_2023.csv" → league="Premier League", season=2023
        # We use if-elif statements to filter out the league name based on keywords in the filename.
        if 'premier' in filename.lower(): # If premier is in the filename:
            df['league'] = 'Premier League' # Then we set the league column to Premier League
        elif 'la_liga' in filename.lower() or 'laliga' in filename.lower():
            df['league'] = 'La Liga'
        elif 'serie' in filename.lower():
            df['league'] = 'Serie A'
        elif 'bundesliga' in filename.lower():
            df['league'] = 'Bundesliga'
        elif 'ligue' in filename.lower():
            df['league'] = 'Ligue 1'
        else:
            df['league'] = 'Unknown' # This is just in case none of the above match, but it shouldn't theoretically happen.
        
        all_transfers.append(df) # For each respective 35 dataframes, we add it to our list of all transfers.
        print(f"   ✓ Loaded {filename}: {len(df)} transfers") # We print a message to inform the user that the file was loaded successfully.
        # We also print how many transfers were in that file.

    except Exception as e: # If for any reason we can't load a file, we catch the error and print a message.
        print(f"   ✗ Failed to load {file}: {e}")

# Combine all dataframes into one big dataframe # transfers will be our final dataframe containing all transfer data.
if all_transfers: # If the list is not empty, i.e., at least one file was loaded successfully.
    transfers = pd.concat(all_transfers, ignore_index=True) #pd.contact allows us to stack multiple dataframes on top of each other.
    #ignore_index=True means we want to create new row indices for the combined dataframe.
    print(f"\n✅ Total transfers loaded: {len(transfers)}") #inform the user of the total number of transfers loaded.
else: # If the list is empty, i.e., no files were loaded successfully.
    print("\n❌ No data loaded!")
    exit() # Exit the script as there's no data to work with.

# ============================================

print("\n🔍 Applying filters...")

# Although the csv files were pre-filtered, we apply additional filters here to ensure data quality.
transfers_filtered = transfers[ # We go to the final dataframe transfers and apply filters to create a new dataframe transfers_filtered.
    (transfers['Transfer_Fee'] >= 5_000_000) &  # €5M minimum minimum transfer fee
    (transfers['Position'].str.contains('Midfield|Attack|Forward|Winger|Striker', na=False, case=False)) # Only attackers and midfielders
]

print(f"✅ After filtering: {len(transfers_filtered)} transfers") # Inform the user how many transfers remain after filtering.
print(f"   Criteria: €5M+ fee, attackers/midfielders only")

# ============================================

# The purpose of this section is to give the user a quick overview of the data collected. Solely for informational purposes.
print("\n" + "="*60) 
print("📊 SUMMARY STATISTICS")
print("="*60)

print(f"\nTotal transfers collected: {len(transfers)}")
print(f"After filtering: {len(transfers_filtered)}")

print("\n📈 Transfers by league:")
print(transfers_filtered['league'].value_counts().to_string()) 
# We go to our filtered dataframe transfers_filtered, select the 'league' column
# value_counts() is to count how many transfers are in each league. to_string() is to make it look nice when printed.

print("\n📈 Transfers by position:")
print(transfers_filtered['Position'].value_counts().head(10).to_string())
# We do the same for the postions. head(10) is to only show the top 10 most common positions.

# mean() gives us the mean for our dataframe.
print(f"\n💰 Average transfer fee: €{transfers_filtered['Transfer_Fee'].mean():,.0f}") 
print(f"💰 Median transfer fee: €{transfers_filtered['Transfer_Fee'].median():,.0f}")
print(f"💰 Highest transfer fee: €{transfers_filtered['Transfer_Fee'].max():,.0f}")


# ============================================


print("\n" + "="*60)
print("📋 SAMPLE OF TOP TRANSFERS")
print("="*60)

# Show top 20 by transfer fee as the user can find that interesting. These can be considered "outliers" in the dataset.
top_transfers = transfers_filtered.nlargest(20, 'Transfer_Fee') # nlargest allows us to get the top 20 rows with the largest values in the 'Transfer_Fee' column.

for i, (idx, transfer) in enumerate(top_transfers.iterrows(), 1): #iterrows() allows us to loop through each row in the dataframe.
    # enumerate() allows us to get a counter (i) over something we are looping through, starting from 1 here.
    # For each row, we get the index (idx) and the row data itself (transfer).
    print(f"\n{i}. {transfer['Player']} ({transfer['Age']} years old)") # For a given row, we print the player's name and age.
    print(f"   Position: {transfer['Position']}")
    print(f"   From: {transfer['Previous_Club']}")
    print(f"   Transfer Fee: €{transfer['Transfer_Fee']:,.0f}")
    print(f"   League: {transfer['league']}")
# We subsequently print the position, previous club, transfer fee, and league for each of these transfers.


# ============================================

# We first save the combined datafram into a new csv file for future use via pandas' to_csv() function.
# This will be saved in the data/ folder.

# Remove any completely empty columns before saving
transfers = transfers.dropna(axis=1, how='all')
transfers_filtered = transfers_filtered.dropna(axis=1, how='all')

transfers.to_csv('data/all_transfers_combined.csv', index=False) #index=False means we don't want to save the row indices in the csv file.
print(f"\n✅ All transfers saved to: data/all_transfers_combined.csv")

#We then do the same for the filtered dataframe in case there are filtered out transfers that were neglected in the initial csv files.
transfers_filtered.to_csv('data/transfers_filtered.csv', index=False)
print(f"✅ Filtered transfers saved to: data/transfers_filtered.csv")

print("\n" + "="*60)
print("✅ DATA COLLECTION COMPLETE!")
print("="*60)
print(f"\nYou have {len(transfers_filtered)} transfers ready for analysis!")
# This gives the full amount of transfers ready for analysis after filtering.
