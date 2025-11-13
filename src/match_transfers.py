import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from glob import glob

# ============================================
print("="*60)
print("MATCHING TRANSFERMARKT & FBREF DATA")
print("TWO-PASS STRATEGY: BEFORE + AFTER")
print("="*60)

# Load Transfermarkt data
print("\n📂 Loading Transfermarkt data...")
transfers = pd.read_csv('data/transfers_filtered.csv')
print(f"✅ Loaded {len(transfers)} transfers")

# ============================================
print("\n📂 Loading ALL FBref data...")

fbref_files = glob('data/fbref/*.csv')
fbref_data = []

for file in fbref_files:
    try:
        df = pd.read_csv(file)
        
        # Extract league and season from filename
        filename = file.split('/')[-1].replace('.csv', '')
        
        if 'premier' in filename.lower():
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
        if len(season_parts) == 2:
            df['fbref_season'] = int(season_parts[0])
        
        fbref_data.append(df)
        print(f"   ✓ Loaded {file}")
    except Exception as e:
        print(f"   ✗ Failed to load {file}: {e}")

fbref = pd.concat(fbref_data, ignore_index=True)
print(f"✅ Total FBref records: {len(fbref)}")

# ============================================
print("\n🔧 Preprocessing data...")

# Standardize player names
transfers['player_clean'] = transfers['Player'].str.lower().str.strip()
fbref['player_clean'] = fbref['Player'].str.lower().str.strip()

# Extract transfer year
transfers['transfer_year'] = transfers['source_file'].str.extract(r'(\d{4})').astype(int)

# Calculate before and after seasons
transfers['season_before'] = transfers['transfer_year'] - 1
transfers['season_after'] = transfers['transfer_year']

# Standardize league names
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

transfers['league_clean'] = transfers['league'].apply(standardize_league)

print("✅ Preprocessing complete")

# Clean up any NaN values in player names
transfers = transfers[transfers['player_clean'].notna()]
fbref = fbref[fbref['player_clean'].notna()]

print(f"✅ After cleaning: {len(transfers)} transfers, {len(fbref)} FBref records")

# ============================================
print("\n🔗 PASS 1: Matching BEFORE season stats...")

def find_best_match(name, choices, threshold=85):
    """Find best matching name using fuzzy string matching"""
    # Handle missing/invalid names
    if pd.isna(name) or not isinstance(name, str) or len(choices) == 0:
        return None, 0
    
    # Filter out any NaN values from choices
    valid_choices = [c for c in choices if pd.notna(c) and isinstance(c, str)]
    
    if len(valid_choices) == 0:
        return None, 0
    
    result = process.extractOne(name, valid_choices, scorer=fuzz.ratio)
    if result and result[1] >= threshold:
        return result[0], result[1]
    return None, 0

before_matches = {}

for idx, transfer in transfers.iterrows():
    player_name = transfer['player_clean']
    dest_league = transfer['league_clean']
    season_before = transfer['season_before']
    
    # Filter FBref for destination league and season before transfer
    fbref_filtered = fbref[
        (fbref['fbref_league'] == dest_league) & 
        (fbref['fbref_season'] == season_before)
    ]
    
    if len(fbref_filtered) > 0:
        fbref_names = fbref_filtered['player_clean'].unique()
        best_match, score = find_best_match(player_name, fbref_names)
        
        if best_match:
            fbref_stats = fbref_filtered[fbref_filtered['player_clean'] == best_match].iloc[0]
            before_matches[idx] = fbref_stats.to_dict()
    
    if (idx + 1) % 50 == 0:
        print(f"   Processed {idx + 1}/{len(transfers)} transfers...")

print(f"✅ BEFORE stats matched: {len(before_matches)}/{len(transfers)} ({len(before_matches)/len(transfers)*100:.1f}%)")

# ============================================
print("\n🔗 PASS 2: Matching AFTER season stats...")

after_matches = {}

for idx, transfer in transfers.iterrows():
    player_name = transfer['player_clean']
    dest_league = transfer['league_clean']
    season_after = transfer['season_after']
    
    # Filter FBref for destination league and season after transfer
    fbref_filtered = fbref[
        (fbref['fbref_league'] == dest_league) & 
        (fbref['fbref_season'] == season_after)
    ]
    
    if len(fbref_filtered) > 0:
        fbref_names = fbref_filtered['player_clean'].unique()
        best_match, score = find_best_match(player_name, fbref_names)
        
        if best_match:
            fbref_stats = fbref_filtered[fbref_filtered['player_clean'] == best_match].iloc[0]
            after_matches[idx] = fbref_stats.to_dict()
    
    if (idx + 1) % 50 == 0:
        print(f"   Processed {idx + 1}/{len(transfers)} transfers...")

print(f"✅ AFTER stats matched: {len(after_matches)}/{len(transfers)} ({len(after_matches)/len(transfers)*100:.1f}%)")

# ============================================
print("\n📊 Creating final datasets...")

# Dataset 1: Players with BOTH before AND after stats (complete comparison)
complete_matches = []
for idx in before_matches.keys():
    if idx in after_matches:
        transfer = transfers.loc[idx]  # Changed from .iloc to .loc
        before_stats = before_matches[idx]
        after_stats = after_matches[idx]
        
        complete_record = {
            **transfer.to_dict(),
            **{f'before_{k}': v for k, v in before_stats.items()},
            **{f'after_{k}': v for k, v in after_stats.items()}
        }
        complete_matches.append(complete_record)

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
