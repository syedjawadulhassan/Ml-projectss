def explain_cluster(cluster_id):
    mapping = {
        0: "High Value Customers",
        1: "Regular Customers",
        2: "Occasional Buyers",
        3: "Low Engagement Customers"
    }
    return mapping.get(cluster_id, "Unknown Segment")