from pymilvus import connections, Collection

connections.connect(alias="default", uri="tcp://localhost:19530")
coll = Collection("collection_hardware")

file_path = r"XXX"
escaped = file_path.replace("\\", "\\\\")

expr = f"source == '{escaped}'"
print("Using expr:", expr)

res = coll.query(
    expr=expr,
    output_fields=["source"],
    limit=1
)
print(res)

if res:
    print("File is indexed:", res[0]["source"])
else:
    print("File not found in collection")
