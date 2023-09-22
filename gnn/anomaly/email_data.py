'''
Data
'''
from pymongo import MongoClient
import networkx as nx

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['123-shield']
collection = db['email_message']

# Create a NetworkX graph
G = nx.Graph()

# Query the MongoDB collection and add data to the graph
for doc in collection.find():
    sender = doc['from']
    recipients = doc['to'] + doc['cc'] + doc['bcc']
    for recipient in recipients:
        if not G.has_edge(sender, recipient):
            G.add_edge(sender, recipient, weight=0)
        G[sender][recipient]['weight'] += 1

# Your graph G is now ready for feature engineering and GNN training
print()
nx.write_graphml(G, 'graph.graphml')
nx.write_gexf(G, 'graph.gexf')

