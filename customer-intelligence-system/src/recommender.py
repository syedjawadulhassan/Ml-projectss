def recommend_products(rules, product):
    product = product.lower().strip()

    # case insensitive match
    filtered = rules[
        rules["antecedents"].str.lower().str.contains(product, na=False)
    ]

    if len(filtered) == 0:
        return filtered

    # sort by strongest rule
    filtered = filtered.sort_values(
        by=["lift", "confidence"],
        ascending=False
    )

    return filtered[[
        "antecedents",
        "consequents",
        "confidence",
        "lift"
    ]]