from pymilvus import connections, utility

connections.connect(alias="default", host="localhost", port=19530)

collection_dp = "collection_hardware"
if utility.has_collection(collection_dp):
    utility.drop_collection(collection_dp)
    print(f"Dropped collection: {collection_dp}")
else:
    print(f"Collection not found: {collection_dp}")
