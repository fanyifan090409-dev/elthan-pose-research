import json
import subprocess

API = "http://127.0.0.1:8080/api"
SU = "ethan"
PW = "isef2027"


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def curl(method, url, data=None, tok=None):
    cmd = ["curl", "-s", "-m", "60", "-X", method, url]
    if tok:
        cmd += ["-H", f"Authorization: Token {tok}"]
    if data is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(data)]
    r = run(cmd)
    try:
        return r.returncode, json.loads(r.stdout) if r.stdout.strip() else None
    except Exception:
        return r.returncode, {"raw": r.stdout[:300]}


def main():
    rc, resp = curl("POST", f"{API}/auth/login", {"username": SU, "password": PW})
    tok = resp["key"]

    rc, proj = curl("GET", f"{API}/projects/1", tok=tok)
    labels_url = proj["labels"]["url"]
    rc, labels = curl("GET", labels_url, tok=tok)
    labels = labels["results"] if isinstance(labels, dict) and "results" in labels else labels
    old_svg = labels[0]["svg"]
    print("old edge count:", old_svg.count("data-node-from"))
    print("has node 17:", 'data-node-to="17"' in old_svg)

    edges0 = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
              [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
              [1, 3], [2, 4], [3, 5], [4, 6]]
    lines = "".join(
        f'<line x1="0" y1="0" x2="1" y2="1" stroke="black" stroke-width="0.5" '
        f'data-node-from="{a}" data-node-to="{b}" />' for a, b in edges0)

    for lab in labels:
        if lab.get("type") == "skeleton":
            lab["svg"] = lines

    rc, resp = curl("PATCH", f"{API}/projects/1", {"labels": labels}, tok=tok)
    print("patch rc:", rc)

    rc, proj2 = curl("GET", f"{API}/projects/1", tok=tok)
    rc, labels2 = curl("GET", proj2["labels"]["url"], tok=tok)
    labels2 = labels2["results"] if isinstance(labels2, dict) and "results" in labels2 else labels2
    svg = labels2[0]["svg"]
    print("new edge count:", svg.count("data-node-from"))
    print("max node idx:", max(int(x) for x in
          __import__("re").findall(r'data-node-(?:from|to)="(\d+)"', svg)))
    print("svg:", svg[:200], "...")


if __name__ == "__main__":
    main()
