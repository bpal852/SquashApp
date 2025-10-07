# -*- coding: utf-8 -*-
"""
Balance 50 squash players into 4 teams (2×13 and 2×12 players), 
with fixed captains, minimizing difference in total team rating,
then print each team's roster sorted by rating in 4 aligned columns.
"""

# --- Data & setup ---
from collections import defaultdict

players = [
    ("Chiu Ho Fai", 9220),
    ("Ashray Ohri", 8500),
    ("Andy Maruca", 8436),
    ("Rishad Schaefer", 4341),
    ("Yeshan Ekanayake", 3993),
    ("Steve West", 3777),
    ("Rob Cranston", 3378),
    ("Chris Skinn", 2800),
    ("Frank Achouch", 2750),
    ("Gareth Janes", 2712),
    ("Jess Wong", 2600),
    ("Russ Lamb", 2500),
    ("Jules Achouch", 2400),
    ("Barry Caveney", 2325),
    ("Dan Ternes", 2101),
    ("Eric Fourcine", 2100),
    ("Harpreet Greewal", 2100),
    ("Vishal Bhammer", 2054),
    ("Jonathan Heritage", 2000),
    ("Ben Litherland", 1980),
    ("Nick Stearn", 1975),
    ("Paul Denham", 1950),
    ("Marco Hoogendijk", 1948),
    ("Raghav Bhandari", 1921),
    ("Sven Olsen", 1919),
    ("Le Shi", 1900),
    ("Sahil Hathiramani", 1850),
    ("Wilfred Lai", 1839),
    ("Raj Kumar", 1750),
    ("Silvia Leung", 1732),
    ("Michael Poole-Wilson", 1700),
    ("Rob Dickson", 1696),
    ("Sheethal Dalpathraj", 1653),
    ("Josh Ngo", 1549),
    ("Suki Chan", 1493),
    ("Lionel Chow", 1440),
    ("Helen Ho", 1300),
    ("Alvina Kwok", 1228),
    ("Belle Ho", 1200),
    ("Lee Simmons", 1200),
    ("Steve Metcalfe", 1100),
    ("Jamie Warner", 1000),
    ("John Griffiths", 952),
    ("Michael Openshaw", 950),
    ("Henry Vera", 943),
    ("Arthur Koeman", 934),
    ("Alison Cumming", 767),
    ("Richardy Healy", 750),
    ("Anna Cooke", 600),
]

captains = {
    1: "Rob Dickson",
    2: "Raghav Bhandari",
    3: "Andy Maruca",
    4: "Jamie Warner",
}

team_size = {1: 12, 2: 12, 3: 12, 4: 13}

# build lookup for ratings
rating_lookup = {name: rating for name, rating in players}

# --- Greedy team assignment (as before) ---
teams = {i: [captains[i]] for i in range(1, 5)}
team_totals = {i: rating_lookup[captains[i]] for i in range(1, 5)}

# remove captains from the pool
remaining = [(n, r) for n, r in players if n not in captains.values()]
remaining.sort(key=lambda x: x[1], reverse=True)

for name, rating in remaining:
    eligible = [t for t in teams if len(teams[t]) < team_size[t]]
    best = min(eligible, key=lambda t: team_totals[t])
    teams[best].append(name)
    team_totals[best] += rating

# --- Now sort each team's roster by rating descending ---
for t in teams:
    teams[t].sort(key=lambda name: rating_lookup[name], reverse=True)

# --- Print in 4 columns ---
max_len = max(len(roster) for roster in teams.values())
for t in teams:
    # append blanks so each column is the same length
    teams[t] += [""] * (max_len - len(teams[t]))

col_width = 25
fmt = ("{:<" + str(col_width) + "}") * 4

# headers
print(fmt.format("Team 1", "Team 2", "Team 3", "Team 4"))
print("-" * (col_width * 4))

# rows
for row in zip(*(teams[i] for i in range(1, 5))):
    # mark captains inline
    row = tuple(
        f"{name} (Cpt)" if name == captains[idx+1] else name
        for idx, name in enumerate(row)
    )
    print(fmt.format(*row))
