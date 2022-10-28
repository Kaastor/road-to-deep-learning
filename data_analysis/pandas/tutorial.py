import pandas as pd

"""
Creating, Loading & Saving
"""

# DataFrame creation
fruit_sales = pd.DataFrame({
    'Apples': [35, 41],
    'Bananas': [21, 34]
}, index=['2017 Sales', '2018 Sales'])

# Series creation
ingredients = pd.Series(
    ['4 cups', '1 cup', '2 large', '1 can'],
    index=['Flour', 'Milk', 'Eggs', 'Spam'],
    name='Dinner'
)

# Load csv with using first column as Index
reviews = pd.read_csv(
    '../../data/wine/winemag-data_first150k.csv',
    index_col=0
)

# Save df to disk
fruit_sales.to_csv('cows_and_goats.csv')

"""
Indexing, Selecting & Assigning
"""

reviews = pd.read_csv("../../data/wine/winemag-data_first150k.csv'", index_col=0)
pd.set_option("display.max_rows", 5)
desc = reviews.description  # get access to the series via dot operator
desc = reviews['description']  # get access to the series via dot operator
print(type(desc))  # dot is Series type

# loc & iloc are preferred accessors
first_row = reviews.iloc[0]  # select first row in the df
first_col = reviews.iloc[:, 0]  # all rows and first index
first_three_rows_for_first_col = reviews.iloc[:3, 0]
first_three_rows_for_first_col = reviews.iloc[[0, 1, 2], 0]
last_five_rows = reviews.iloc[-5:]
specific_rows = sample_reviews = reviews.iloc[[1, 2, 3, 5, 8]]

first_entry = reviews.loc[0, 'country']
specific_columns_by_name = reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
specific_rows_columns_by_name = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]

"""
iloc indexing - 0:10 will select entries 0,...,9
loc indexing - 0:10 will select entries 0,...,10
"""

first_100_rows_for_two_cols = reviews.loc[:99, ['country', 'variety']]

# Change index for the df (title is a column)
reviews.set_index("title")

series = reviews.country == 'Italy'  # Series containing True or False values for every row
select_rows_with_italy_wines = reviews.loc[reviews.country == 'Italy']
points_and_in_two_countries = top_oceania_wines = reviews.loc[
    (reviews.points >= 95) & reviews.country.isin(['Australia', 'New Zealand'])
    ]

"""
Summary Functions and Maps
"""

stats_about_column = reviews.points.describe()
column_mean = reviews.points.mean()
reviews_per_country = reviews.country.value_counts()

reviews_price_mean = reviews.points.mean()
centered_price = reviews.price.map(lambda p: p - reviews_price_mean)  # map every value in series to another val


# apply function to whole df (for each row)
def remean_points(row):
    row.points = row.points - reviews_price_mean
    return row


reviews.apply(remean_points, axis='columns')  # axis='index' would transform each column

# faster than map() or apply() because uses speedups
review_points_mean = reviews.points.mean()
remeaned = reviews.points - review_points_mean

# Create a variable bargain_wine with the title of the wine with the highest points-to-price ratio in the dataset.
index = (reviews.points / reviews.price).argmax()  # get index of the max ratio
bargain_wine = reviews.loc[index, 'title']  # use index to fetch title of wine

"""
Grouping and Sorting
"""

reviews.groupby('points').points.count()
# pick out the best wine by country and province
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
# statistical summary of the dataset for price
reviews.groupby(['country']).price.agg([len, min, max])
# index by country and province and by value show len of description
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])

# go back to single index (from multi-index)
countries_reviewed.reset_index()
# minimum and maximum prices for each variety of wine
price_extremes = reviews.groupby('variety').price.agg([min, max])
# sort previous df first by min, then max in descending fashion
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)

reviewer_mean_ratings = reviews.groupby('taster_name').points.agg('mean')
# index is a MultiIndex of {country, variety} pairs.
# Sort the values in the Series in descending order based on wine count
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)



