import pandas as pd

"""
Data structure:
url - url
type - 0/1, where 1 means phishing url
"""


def load_data():
    dataset = [
        pd.read_csv('phish_score.csv', delimiter=',')
    ]

    return dataset


load_data()

# TODO: załadować dane 'dobre' (z mongo) i 'złe' (zobaczyć ile ich jest i tyle pobrać dobrych).
#  Zbudować generatory.
#  Zobaczyć czy model nauczy się klasyfikować
