import json
import subprocess
import time

API = "http://127.0.0.1:8080/api"
SU, PW = "ethan", "isef2027-anno"


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def curl_json(method, url, data=None, auth=None):
    cmd = ["curl", "-s", "-m", "120", "-X", method, url]
    if auth:
        cmd += ["-H", f"Authorization: Token {auth}"]
    if data is not None:
        cmd += ["-H", "Content-Type: application/json",
                "-d", json.dumps(data, ensure_ascii=False)]
    r = run(cmd)
    try:
        return r.returncode, json.loads(r.stdout) if r.stdout.strip() else None
    except Exception:
        return r.returncode, {"raw": r.stdout[:300], "err": r.stderr[:300]}


rc, resp = curl_json("POST", f"{API}/auth/login",
                     {"username": SU, "password": PW})
tok = resp["key"]

ready = False
for _ in range(40):
    rc, t = curl_json("GET", f"{API}/tasks/1", auth=tok)
    print("task1 poll:", t.get("status"), "size:", t.get("size"),
          "proj:", t.get("project_id"))
    if t and t.get("size") == 60:
        ready = True
        break
    time.sleep(5)
print("import ready:", ready)

rc, t = curl_json("PATCH", f"{API}/tasks/1", {"project_id": 1}, auth=tok)
print("move to project 1 rc:", rc, "proj:", t.get("project_id") if t else None)

rc, _ = curl_json("DELETE", f"{API}/projects/2", auth=tok)
print("delete project 2 rc:", rc)

rc, projs = curl_json("GET", f"{API}/projects", auth=tok)
print("projects:", [(p["id"], p["name"]) for p in (projs or [])])

rc, t = curl_json("GET", f"{API}/tasks/1", auth=tok)
print("task1 final:", t.get("status"), "size:", t.get("size"),
      "proj:", t.get("project_id"), "frames:", t.get("size"))
