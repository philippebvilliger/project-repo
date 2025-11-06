import pandas as pd # We import pandas to handle dataframes
import time # We import time to add delays between requests
import random # We import random to add randomness to delay times
import requests # We use requests to fetch pages and maintain a session to mimic a browser

# ============================================


print("="*60)
print("FBREF PERFORMANCE DATA COLLECTION")
print("="*60)

# We're gonna create a dictionary that maps league names to their FBref competition IDs
# These IDs are found in the URLs of the league pages on FBref
fbref_leagues = {
    'Premier-League': 9,      # English Premier League
    'La-Liga': 12,            # Spanish La Liga
    'Serie-A': 11,            # Italian Serie A
    'Bundesliga': 20,         # German Bundesliga
    'Ligue-1': 13             # French Ligue 1
}

# We take the seasons we want to collect data for.
# We also collect data for 2016-2017 as the whole purpose is to analyze performance before and after the transfer.
# Format: 'YYYY-YYYY' (e.g., '2017-2018')
seasons = [
    '2016-2017', '2017-2018', '2018-2019', '2019-2020',
    '2020-2021', '2021-2022', '2022-2023', '2023-2024'
]

# We create an empty list to store all the dataframes we collect. One dataframe per league, per season.
all_stats = []

print(f"\n📊 Collecting data for {len(fbref_leagues)} leagues × {len(seasons)} seasons")
print(f"   Total: {len(fbref_leagues) * len(seasons)} datasets to collect\n") # Informs the user of the amount of dataframes to be collected

# We create a session object which allows us to persist certain parameters across requests.
# Normally, when you make a request using requests.get(), it's a one-off request and it resets everything after.
# But with a session, we can maintain cookies and headers across multiple requests.
# This makes our scraping more efficient and consistent, as the same “browser identity” is reused for every page we request.
session = requests.Session()
# Headers basically tell the website WHO you are and WHAT you want.
# We update headers to make our HTTP requests look like they’re coming from a real web browser, rather than a script or bot
session.headers.update({ 
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36',
    # Tells the website what kind of browser and operating system you are using.
    'Accept-Language': 'en-US,en;q=0.9', # Prefers English language content
    'Accept-Encoding': 'gzip, deflate, br', # Tells the server what kind of compression we can handle
    'Referer': 'https://fbref.com/', # Tells the server where we came from (the homepage in this case)
    'DNT': '1', # Do Not Track request 
    'Connection': 'keep-alive', # Keep the connection open for multiple requests
})

# Loop through each season
for season in seasons: # We loop through each season starting from 2016-2017 
    # We loop through each league name and its corresponding FBref ID thanks to items() that retrieves the key-value pairs of fbref_leagues
    for league_name, league_id in fbref_leagues.items():
        
        # Construct the URL for this specific league and season
        # Example: https://fbref.com/en/comps/9/2023-2024/stats/2023-2024-Premier-League-Stats
        url = f'https://fbref.com/en/comps/{league_id}/{season}/stats/{season}-{league_name}-Stats' # We create the URL thanks to an f string
        
        try: # We use try-except to handle any errors that may occur during data fetching e.g., network issues, invalid URLs...
            # sleep() allows us to pause execution for a given number of seconds.
            # We do this here to avoid overwhelming the FBref server with requests.
            time.sleep(random.uniform(3, 6))
            
            print(f"Fetching {league_name} {season}...", end=" ")
            
            # Try a few times in case of intermittent blocks/connection issues
            tables = None # We initialize the variable that will hold the tables we scrape from the page
            last_error = None # Every time an attempt fails (network issue, timeout, HTTP 403 ...), Python catches the exception and stores it in last_error
            for attempt in range(1, 4): # We want to try to fetch the data from the page up to 3 times before giving up
                try: # We use another try-except block to catch errors during the actual request
                    resp = session.get(url, timeout=30) # We use the session object to make a GET request to the URL with a timeout of 30 seconds
                    resp.raise_for_status() # This raises an HTTPError if the HTTP request returned an unsuccessful status code (e.g., 404, 500...)
                    tables = pd.read_html(resp.text) # Pandas is used to read all HTML tables from the page content
                    last_error = None # If we reach this point, it means the request was successful, so we reset last_error
                    break # Exit the loop
                except Exception as attempt_err: # If an error occurs during the request or reading tables, we catch it here
                    last_error = attempt_err # We store this exception in the last_error variable
                    # small wait before retrying
                    time.sleep(2 + attempt) 
            
            if tables is None: # After 3 attempts, if we still have no tables, we raise the last error encountered
                raise last_error
            # The script stops and shows what went wrong

            # The main stats table is usually the first table on the page. This is the table of interest for us.
            stats = tables[0]
            
            # Add metadata columns to track where this data came from
            # This is crucial for later when we combine all data as we need to know what season/league each row belongs to
            stats['season'] = season           # Which season (e.g., '2023-2024')
            stats['league'] = league_name      # Which league (e.g., 'Premier-League')
            
            # We add this main stats table to our list of all stats.
            # It's a loop so we will be adding this for each league and each season.
            all_stats.append(stats)
            
            # Print success message with number of players found, once it's done.
            print(f"✓ ({len(stats)} players)")
            
        except Exception as e:  # As mentioned before, if something goes wrong (bad URL, connection issue, etc.), print error
            print(f"✗ Error: {e}")

# ============================================


if all_stats: # If the list is not empty, we proceed to combine all dataframes
    print("\n" + "="*60)
    print("📊 COMBINING ALL DATA")
    print("="*60)
    
    # Combine all DataFrames in the list into ONE big DataFrame
    # pd.concat() stacks DataFrames vertically (puts them on top of each other)
    # ignore_index=True means create new row numbers (0, 1, 2, ...) instead of keeping old ones that were from individual DataFrames
    # This prevents duplicate index numbers
    fbref_stats = pd.concat(all_stats, ignore_index=True)
    
    print(f"\n✅ Total player-season records collected: {len(fbref_stats)}")
    #len(fbref_stats) = total number of rows in the combined DataFrame
    
    # Show sample of what we collected
    print("\n📋 Sample of collected data:")
    print(fbref_stats.head(10)) #.head(10) shows the first 10 rows of the new DataFrame created
    
    print("\n📊 Columns available:")
    print(fbref_stats.columns.tolist()) # columns gives us all column names fromt the new fbref_stats DataFrame created
    # .tolist() converts to a regular Python list for easier reading
    
    # Save to CSV file
    output_file = 'data/raw/fbref_stats_raw.csv' # This is the file path where we will save the data
    fbref_stats.to_csv(output_file, index=False) # to_csv() saves the new DataFrame to a CSV file
    # index=False means do NOT save the row numbers (0, 1, 2, ...) to the file
    
    print(f"\n✅ Data saved to: {output_file}") # to inform the user where the data was saved
    print(f"   Total records: {len(fbref_stats)}") # to inform the user of the total number of records saved
    
    # Show breakdown by league
    print("\n📈 Records by league:")
    print(fbref_stats['league'].value_counts().to_string()) # .value_counts() counts how many times each unique value appears in the 'league' column
    # .to_string() formats it nicely for printing

    # Show breakdown by season
    print("\n📈 Records by season:")
    print(fbref_stats['season'].value_counts().to_string()) # Same as above but for 'season' column.
    
else: # if the list is empty and we haven't collected any data for any league/season, print error.
    print("\n❌ No data collected!")
    print("   Check your internet connection or FBref website availability.")

print("\n" + "="*60)
print("✅ FBREF DATA COLLECTION COMPLETE!")
print("="*60)

