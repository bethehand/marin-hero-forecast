"""Minimal read-only client for the public W&B project marin-community/marin_moe (anonymous GraphQL)."""
import json, time, urllib.request

EP = "https://api.wandb.ai/graphql"


def gql(query, variables, retries=4):
    body = json.dumps({"query": query, "variables": variables}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(EP, data=body, headers={"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=300).read())
            if "errors" in out:
                raise RuntimeError(out["errors"])
            return out["data"]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def run(name, fields="name displayName state createdAt config summaryMetrics"):
    q = 'query($n: String!) { project(name: "marin_moe", entityName: "marin-community") { run(name: $n) { %s } } }' % fields
    return gql(q, {"n": name})["project"]["run"]


def runs(regex, fields="name displayName state createdAt config summaryMetrics", page=50):
    """All runs whose display name matches regex, newest first."""
    q = """query($f: JSONString, $after: String, $first: Int) { project(name: "marin_moe", entityName: "marin-community") {
      runs(filters: $f, after: $after, first: $first, order: "-created_at") {
        pageInfo { endCursor hasNextPage } edges { node { %s } } } } }""" % fields
    f = json.dumps({"display_name": {"$regex": regex}})
    after, out = None, []
    while True:
        r = gql(q, {"f": f, "after": after, "first": page})["project"]["runs"]
        out += [e["node"] for e in r["edges"]]
        if not r["pageInfo"]["hasNextPage"]:
            return out
        after = r["pageInfo"]["endCursor"]
