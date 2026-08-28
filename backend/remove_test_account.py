import json

path = "data/db.json"
target_email = "auth-fix-test-20260821@example.com"

with open(path, "r") as f:
    data = json.load(f)

users = data.get("users", [])
match = [u for u in users if u.get("email") == target_email]
if not match:
    print("No matching user found; nothing to do.")
else:
    user_id = match[0]["_id"]
    data["users"] = [u for u in users if u.get("email") != target_email]
    removed_related = 0
    for coll in ("datasets", "models", "reports"):
        items = data.get(coll, [])
        before = len(items)
        data[coll] = [d for d in items if d.get("user_id") != user_id]
        removed_related += before - len(data[coll])
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Removed user {target_email} (id={user_id}) and {removed_related} related record(s).")
