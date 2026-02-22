import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def _clean_itemset(x):
    # handles both frozenset and already-string cases
    if isinstance(x, (set, frozenset, list, tuple)):
        return ", ".join(sorted(list(x)))
    return str(x)

def market_basket(df):
    # Create basket matrix
    basket = (
        df.groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum()
        .unstack()
        .fillna(0)
    )

    # Convert to binary
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Frequent itemsets
    freq_items = apriori(
        basket,
        min_support=0.02,
        use_colnames=True
    )

    # Association rules
    rules = association_rules(
        freq_items,
        metric="lift",
        min_threshold=1
    )

    # ✅ HARD CLEAN (never shows frozenset)
    rules["antecedents"] = rules["antecedents"].apply(_clean_itemset)
    rules["consequents"] = rules["consequents"].apply(_clean_itemset)

    # Sort strongest first
    rules = rules.sort_values(
        by=["lift", "confidence"],
        ascending=False
    ).reset_index(drop=True)

    return rules