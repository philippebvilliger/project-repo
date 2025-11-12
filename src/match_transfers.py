import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

# ============================================
# MATCH TRANSFERS WITH FBREF PERFORMANCE DATA
# ============================================

print("="*80)
print("MATCHING TRANSFERS WITH PERFORMANCE DATA")
print("="*80)

# ============================================
# STEP 1: LOAD DATA
# ============================================

print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load transfers
print("\n📂 Loading Transfermarkt data...")
transfers = pd.read_csv('data/transfers_filtered.csv')
print(f"✅ Loaded {len(transfers)} transfers")

# Load FBref stats
print("\n📂 Loading FBref data...")
fbref = pd.read_csv('data/processed/fbref_cleaned.csv')
print(f"✅ Loaded {len(fbref)} player-season records")

# ============================================
# STEP 2: PARSE TRANSFER YEAR FROM source_file
# ============================================

print("\n" + "="*80)
print("STEP 2: EXTRACTING TRANSFER YEARS")
print("="*80)

def extract_year_from_filename(filename):
    """Extract year from filename like 'Bundesliga_2023.csv' or 'premier_league_2023.csv'"""
    if pd.isna(filename):
        return None
    # Extract 4-digit year from filename
    match = re.search(r'(\d{4})', str(filename))
    if match:
        return int(match.group(1))
    return None

transfers['transfer_year'] = transfers['source_file'].apply(extract_year_from_filename)

print(f"\n✅ Extracted transfer years")
print(f"   Years range: {transfers['transfer_year'].min()} - {transfers['transfer_year'].max()}")
print(f"   Missing years: {transfers['transfer_year'].isna().sum()}")

# Remove transfers without year
before_drop = len(transfers)
transfers = transfers[transfers['transfer_year'].notna()].copy()
after_drop = len(transfers)
if before_drop != after_drop:
    print(f"   Removed {before_drop - after_drop} transfers without year")

# ============================================
# STEP 3: NAME STANDARDIZATION
# ============================================

print("\n" + "="*80)
print("STEP 3: STANDARDIZING PLAYER NAMES")
print("="*80)

def standardize_name(name):
    """Standardize player names for matching"""
    if pd.isna(name):
        return ""
    
    name = str(name).lower().strip()
    
    # Remove accents and special characters
    replacements = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a', 'ä': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o', 'ö': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ñ': 'n', 'ç': 'c', 'ć': 'c', 'č': 'c',
        'ş': 's', 'š': 's', 'ž': 'z', 'đ': 'd',
        'ø': 'o', 'å': 'a', 'æ': 'ae'
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove common suffixes/prefixes
    name = re.sub(r'\s+(jr|sr|ii|iii|iv)\.?$', '', name)
    
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip()

# Standardize names
transfers['player_std'] = transfers['Player'].apply(standardize_name)
fbref['player_std'] = fbref['Player'].apply(standardize_name)

print(f"✅ Standardized names")
print(f"   Sample transfer names: {transfers['player_std'].head(3).tolist()}")
print(f"   Sample FBref names: {fbref['player_std'].head(3).tolist()}")

# ============================================
# STEP 4: LEAGUE STANDARDIZATION
# ============================================

print("\n" + "="*80)
print("STEP 4: STANDARDIZING LEAGUE NAMES")
print("="*80)

# Standardize league names
league_mapping = {
    'bundesliga': 'Bundesliga',
    'la liga': 'La-Liga',
    'serie a': 'Serie-A',
    'premier league': 'Premier-League',
    'ligue 1': 'Ligue-1'
}

def standardize_league(league):
    """Standardize league names"""
    if pd.isna(league):
        return None
    league_lower = str(league).lower().strip()
    for key, value in league_mapping.items():
        if key in league_lower:
            return value
    return league

transfers['league_std'] = transfers['league'].apply(standardize_league)
fbref['league_std'] = fbref['league'].apply(standardize_league)

print(f"✅ Standardized leagues")
print(f"   Transfer leagues: {transfers['league_std'].unique()}")
print(f"   FBref leagues: {fbref['league_std'].unique()}")

# ============================================
# STEP 5: CREATE SEASON STRINGS
# ============================================

print("\n" + "="*80)
print("STEP 5: MAPPING TRANSFER YEARS TO SEASONS")
print("="*80)

def year_to_season(year):
    """Convert transfer year to season format"""
    # If transfer in 2023, the season before is 2022-2023, after is 2023-2024
    season_before = f"{year-1}-{year}"
    season_after = f"{year}-{year+1}"
    return season_before, season_after

# Add season columns to transfers
transfers[['season_before', 'season_after']] = transfers['transfer_year'].apply(
    lambda x: pd.Series(year_to_season(x))
)

print(f"✅ Created season mappings")
print(f"\n   Sample mappings:")
for i in range(min(3, len(transfers))):
    print(f"   Transfer year {transfers.iloc[i]['transfer_year']} → "
          f"Before: {transfers.iloc[i]['season_before']}, "
          f"After: {transfers.iloc[i]['season_after']}")

# ============================================
# STEP 6: FUZZY NAME MATCHING FUNCTION
# ============================================

def fuzzy_match_name(name1, name2, threshold=0.85):
    """Check if two names are similar enough to be considered a match"""
    return SequenceMatcher(None, name1, name2).ratio() >= threshold

# ============================================
# STEP 7: MATCH TRANSFERS WITH PERFORMANCE
# ============================================

print("\n" + "="*80)
print("STEP 7: MATCHING TRANSFERS WITH PERFORMANCE DATA")
print("="*80)

matched_data = []
unmatched_transfers = []

for idx, transfer in transfers.iterrows():
    player_name = transfer['player_std']
    league = transfer['league_std']
    season_before = transfer['season_before']
    season_after = transfer['season_after']
    
    if (idx + 1) % 100 == 0:
        print(f"   Processing transfer {idx + 1}/{len(transfers)}...")
    
    # Find player in FBref for season BEFORE transfer
    mask_before = (
        (fbref['player_std'] == player_name) &
        (fbref['league_std'] == league) &
        (fbref['season'] == season_before)
    )
    
    stats_before = fbref[mask_before]
    
    # If exact match not found, try fuzzy matching
    if len(stats_before) == 0:
        mask_fuzzy = (
            (fbref['league_std'] == league) &
            (fbref['season'] == season_before)
        )
        potential_matches = fbref[mask_fuzzy]
        
        for _, potential in potential_matches.iterrows():
            if fuzzy_match_name(player_name, potential['player_std']):
                stats_before = pd.DataFrame([potential])
                break
    
    # Find player in FBref for season AFTER transfer
    mask_after = (
        (fbref['player_std'] == player_name) &
        (fbref['league_std'] == league) &
        (fbref['season'] == season_after)
    )
    
    stats_after = fbref[mask_after]
    
    # If exact match not found, try fuzzy matching
    if len(stats_after) == 0:
        mask_fuzzy = (
            (fbref['league_std'] == league) &
            (fbref['season'] == season_after)
        )
        potential_matches = fbref[mask_fuzzy]
        
        for _, potential in potential_matches.iterrows():
            if fuzzy_match_name(player_name, potential['player_std']):
                stats_after = pd.DataFrame([potential])
                break
    
    # Create matched record
    if len(stats_before) > 0 or len(stats_after) > 0:
        # Take first match if multiple
        before_stats = stats_before.iloc[0] if len(stats_before) > 0 else None
        after_stats = stats_after.iloc[0] if len(stats_after) > 0 else None
        
        match_record = {
            # Transfer info
            'player_name': transfer['Player'],
            'age': transfer['Age'],
            'position': transfer['Position'],
            'nationality': transfer['Nationality'],
            'transfer_fee': transfer['Transfer_Fee'],
            'previous_club': transfer['Previous_Club'],
            'market_value': transfer['Market_Value'],
            'transfer_year': transfer['transfer_year'],
            'league': league,
            'season_before': season_before,
            'season_after': season_after,
            
            # Before transfer stats
            'before_squad': before_stats['Squad'] if before_stats is not None else None,
            'before_mp': before_stats['MP'] if before_stats is not None else None,
            'before_starts': before_stats['Starts'] if before_stats is not None else None,
            'before_min': before_stats['Min'] if before_stats is not None else None,
            'before_90s': before_stats['90s'] if before_stats is not None else None,
            'before_gls': before_stats['Gls'] if before_stats is not None else None,
            'before_ast': before_stats['Ast'] if before_stats is not None else None,
            'before_g_pk': before_stats['G-PK'] if before_stats is not None else None,
            'before_pk': before_stats['PK'] if before_stats is not None else None,
            'before_crdy': before_stats['CrdY'] if before_stats is not None else None,
            'before_crdr': before_stats['CrdR'] if before_stats is not None else None,
            'before_gls_per_90': before_stats['Gls_per_90'] if before_stats is not None else None,
            'before_ast_per_90': before_stats['Ast_per_90'] if before_stats is not None else None,
            'before_ga_per_90': before_stats['GA_per_90'] if before_stats is not None else None,
            
            # After transfer stats
            'after_squad': after_stats['Squad'] if after_stats is not None else None,
            'after_mp': after_stats['MP'] if after_stats is not None else None,
            'after_starts': after_stats['Starts'] if after_stats is not None else None,
            'after_min': after_stats['Min'] if after_stats is not None else None,
            'after_90s': after_stats['90s'] if after_stats is not None else None,
            'after_gls': after_stats['Gls'] if after_stats is not None else None,
            'after_ast': after_stats['Ast'] if after_stats is not None else None,
            'after_g_pk': after_stats['G-PK'] if after_stats is not None else None,
            'after_pk': after_stats['PK'] if after_stats is not None else None,
            'after_crdy': after_stats['CrdY'] if after_stats is not None else None,
            'after_crdr': after_stats['CrdR'] if after_stats is not None else None,
            'after_gls_per_90': after_stats['Gls_per_90'] if after_stats is not None else None,
            'after_ast_per_90': after_stats['Ast_per_90'] if after_stats is not None else None,
            'after_ga_per_90': after_stats['GA_per_90'] if after_stats is not None else None,
            
            # Match quality indicators
            'has_before_data': before_stats is not None,
            'has_after_data': after_stats is not None,
            'has_both_data': (before_stats is not None) and (after_stats is not None)
        }
        
        matched_data.append(match_record)
    else:
        unmatched_transfers.append({
            'player_name': transfer['Player'],
            'league': league,
            'transfer_year': transfer['transfer_year']
        })

# ============================================
# STEP 8: CREATE FINAL DATASET
# ============================================

print("\n" + "="*80)
print("STEP 8: CREATING FINAL DATASET")
print("="*80)

matched_df = pd.DataFrame(matched_data)

print(f"\n✅ Matching complete!")
print(f"   Total transfers: {len(transfers)}")
print(f"   Matched with some data: {len(matched_df)}")
print(f"   Matched with BOTH before/after: {matched_df['has_both_data'].sum()}")
print(f"   Matched with only BEFORE: {(matched_df['has_before_data'] & ~matched_df['has_after_data']).sum()}")
print(f"   Matched with only AFTER: {(matched_df['has_after_data'] & ~matched_df['has_before_data']).sum()}")
print(f"   Completely unmatched: {len(unmatched_transfers)}")

# ============================================
# STEP 9: CALCULATE PERFORMANCE CHANGES
# ============================================

print("\n" + "="*80)
print("STEP 9: CALCULATING PERFORMANCE CHANGES")
print("="*80)

# Only calculate for records with both before and after data
complete_data = matched_df[matched_df['has_both_data']].copy()

if len(complete_data) > 0:
    # Calculate changes
    complete_data['change_gls'] = complete_data['after_gls'] - complete_data['before_gls']
    complete_data['change_ast'] = complete_data['after_ast'] - complete_data['before_ast']
    complete_data['change_mp'] = complete_data['after_mp'] - complete_data['before_mp']
    complete_data['change_gls_per_90'] = complete_data['after_gls_per_90'] - complete_data['before_gls_per_90']
    complete_data['change_ast_per_90'] = complete_data['after_ast_per_90'] - complete_data['before_ast_per_90']
    complete_data['change_ga_per_90'] = complete_data['after_ga_per_90'] - complete_data['before_ga_per_90']
    
    print(f"✅ Calculated performance changes for {len(complete_data)} transfers")

# ============================================
# STEP 10: SAVE RESULTS
# ============================================

print("\n" + "="*80)
print("STEP 10: SAVING RESULTS")
print("="*80)

# Save all matched data
matched_df.to_csv('data/processed/transfers_matched_all.csv', index=False)
print(f"\n✅ Saved all matched data: data/processed/transfers_matched_all.csv")
print(f"   Records: {len(matched_df)}")

# Save complete data (with both before and after)
if len(complete_data) > 0:
    complete_data.to_csv('data/processed/transfers_matched_complete.csv', index=False)
    print(f"\n✅ Saved complete data: data/processed/transfers_matched_complete.csv")
    print(f"   Records: {len(complete_data)}")

# Save unmatched transfers for review
if len(unmatched_transfers) > 0:
    unmatched_df = pd.DataFrame(unmatched_transfers)
    unmatched_df.to_csv('data/processed/transfers_unmatched.csv', index=False)
    print(f"\n✅ Saved unmatched transfers: data/processed/transfers_unmatched.csv")
    print(f"   Records: {len(unmatched_df)}")

# ============================================
# STEP 11: SUMMARY STATISTICS
# ============================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

if len(complete_data) > 0:
    print(f"\n📊 Performance Changes (Complete Data Only, n={len(complete_data)}):")
    print(f"\n   Goals:")
    print(f"      Mean change: {complete_data['change_gls'].mean():.2f}")
    print(f"      Median change: {complete_data['change_gls'].median():.2f}")
    print(f"      Improved: {(complete_data['change_gls'] > 0).sum()} ({(complete_data['change_gls'] > 0).sum()/len(complete_data)*100:.1f}%)")
    print(f"      Declined: {(complete_data['change_gls'] < 0).sum()} ({(complete_data['change_gls'] < 0).sum()/len(complete_data)*100:.1f}%)")
    
    print(f"\n   Assists:")
    print(f"      Mean change: {complete_data['change_ast'].mean():.2f}")
    print(f"      Median change: {complete_data['change_ast'].median():.2f}")
    print(f"      Improved: {(complete_data['change_ast'] > 0).sum()} ({(complete_data['change_ast'] > 0).sum()/len(complete_data)*100:.1f}%)")
    print(f"      Declined: {(complete_data['change_ast'] < 0).sum()} ({(complete_data['change_ast'] < 0).sum()/len(complete_data)*100:.1f}%)")
    
    print(f"\n   Goals per 90:")
    print(f"      Mean change: {complete_data['change_gls_per_90'].mean():.3f}")
    print(f"      Median change: {complete_data['change_gls_per_90'].median():.3f}")

# Breakdown by league
print(f"\n📈 Matching success by league:")
for league in matched_df['league'].unique():
    league_data = matched_df[matched_df['league'] == league]
    complete_pct = league_data['has_both_data'].sum() / len(league_data) * 100
    print(f"   {league}: {league_data['has_both_data'].sum()}/{len(league_data)} complete ({complete_pct:.1f}%)")

print("\n" + "="*80)
print("✅ MATCHING COMPLETE!")
print("="*80)

