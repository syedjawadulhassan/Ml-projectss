from src.preprocessing import load_data
from src.rfm_analysis import create_rfm
from src.clustering import perform_clustering
from src.association_rules import market_basket

df = load_data("data/sample_transactions.csv")

rfm = create_rfm(df)
clustered = perform_clustering(rfm)

rules = market_basket(df)

print(clustered.head())
print(rules.head())