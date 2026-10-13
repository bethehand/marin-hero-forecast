"""Read-only anonymous W&B GraphQL helpers for marin-community/marin_moe."""
import json, time, urllib.request
EP = "https://api.wandb.ai/graphql"

def gql(q, v, tries=4):
    body = json.dumps({"query": q, "variables": v}).encode()
    for i in range(tries):
        try:
            out = json.loads(urllib.request.urlopen(urllib.request.Request(EP, data=body, headers={"Content-Type": "application/json"}), timeout=300).read())
            if "errors" in out and not out.get("data"):
                raise RuntimeError(out["errors"])
            return out["data"]
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))

def list_runs(regex, first=100):
    q = """query($f: JSONString!, $first: Int!, $after: String) { project(name: "marin_moe", entityName: "marin-community") {
      runs(filters: $f, first: $first, after: $after) { pageInfo { hasNextPage endCursor }
        edges { node { name displayName state createdAt heartbeatAt group } } } } }"""
    out, after = [], None
    while True:
        d = gql(q, {"f": json.dumps({"displayName": {"$regex": regex}}), "first": first, "after": after})["project"]["runs"]
        out += [e["node"] for e in d["edges"]]
        if not d["pageInfo"]["hasNextPage"]:
            return out
        after = d["pageInfo"]["endCursor"]

def run_meta(name):
    q = """query($n: String!) { project(name: "marin_moe", entityName: "marin-community") {
      run(name: $n) { name displayName state createdAt heartbeatAt config summaryMetrics historyKeys } } }"""
    return gql(q, {"n": name})["project"]["run"]

def history(name, keys, samples=100000):
    q = """query($n: String!, $specs: [JSONString!]!) { project(name: "marin_moe", entityName: "marin-community") {
      run(name: $n) { sampledHistory(specs: $specs) } } }"""
    spec = json.dumps({"keys": ["_step", *keys], "samples": samples})
    rows = gql(q, {"n": name, "specs": [spec]})["project"]["run"]["sampledHistory"][0]
    return sorted(rows, key=lambda r: r["_step"])
