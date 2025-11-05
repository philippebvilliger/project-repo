
import soccerdata as sd
import pandas as pd

tm = sd.Transfermarkt(leagues=['ENG-Premier League', 'ESP-La Liga',  'ITA-Serie A', 'GER-Bundesliga', 'FRA-Ligue 1'])
# tm is a scraper i.e., a program that automatically extracts data from a website. Here the the webiste is transfermarkt.com
# we choose leagues because the website has separate pasges for each league e.g., https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1


transfers = tm.read_transfers(stat_type='permanent', years=range(2017, 2025))
# the function read_transfers will go and extract all transfer data. It's a built-in function in soccerdata
# This should bring us to https://www.transfermarkt.com/serie-a/transfers/wettbewerb/IT1/plus/?saison_id=2023&s_w=&leihe=0&intern=0
#stat_type='permanent' gives us only permanent transfers i.e., we wish to exclude loans
# years=range(2017, 2025) gives us transfers from 2017 to 2024. This corresponds to the dropdown menu on the website.

transfers_filtered = transfers[
    (transfers['fee_cleaned'] >= 5_000_000) &  # We only want transfers with a fee of at least 5 million euros.
    (transfers['position'].str.contains('Midfield|Attack|Forward', na=False)) # we only keep the positions that contain Midfield, Attack or Forward
] #na meaning not available data is false because we have to know the position here
# Python has now extracted the data and begins filtering it based on our criteria.
#Important to note that these criteria can't be selected directly on the website.
# The data extracted is a table which includes the following data columns: player_name, position, age, from, to, fee_cleaned, season 
# you can look at image for example

