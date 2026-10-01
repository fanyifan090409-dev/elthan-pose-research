import json
import subprocess
import time
import zipfile
from pathlib import Path

BASE = Path("/home/ubuntu/isef/data/raw/web-batch1")
IMGROOT = BASE / "images" / "squat"
OUT = BASE / "practice-annot"
ZIP = BASE / "practice-frames.zip"

SU = "elthan"
PW = "__REMOVED__"
EMAIL = "__REMOVED__"
API = "http://127.0.0.1:8080/api"


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def curl_json(method, url, data=None, auth=None):
    cmd = ["curl", "-s", "-m", "120", "-X", method, url]
    if auth:
        cmd += ["-H", f"Authorization: Token {auth}"]
    if data is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(data, ensure_ascii=False)]
    r = run(cmd)
    try:
        return r.returncode, json.loads(r.stdout) if r.stdout.strip() else None
    except Exception:
        return r.returncode, {"raw": r.stdout[:500], "err": r.stderr[:500]}


def make_labels():
    names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
    palette = ["#ff0000", "#ff7f00", "#b2b200", "#00b200", "#00b2b2",
               "#007fff", "#7f00ff", "#ff00ff", "#8c8c8c", "#5c5c5c",
               "#ff4d4d", "#ffb34d", "#e6e64d", "#4dff4d", "#4dd2ff",
               "#4d4dff", "#e64dff"]
    sublabels = [
        {"name": n, "color": c, "type": "points",
         "svg": f'<circle cx="0" cy="0" r="4" stroke="black" fill="{c}" />'}
        for n, c in zip(names, palette)
    ]
    edges = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
             [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
             [1, 3], [2, 4], [3, 5], [4, 6]]
    lines = "".join(
        f'<line x1="0" y1="0" x2="1" y2="1" stroke="black" stroke-width="0.5" '
        f'data-node-from="{a}" data-node-to="{b}" />' for a, b in edges)
    return [{
        "name": "person",
        "color": "#0a66c2",
        "type": "skeleton",
        "attributes": [],
        "sublabels": sublabels,
        "svg": lines,
    }]


def prepare_frames():
    dirs = sorted(IMGROOT.iterdir())
    picked = []
    for want in ("标准", "教学", "错误"):
        for d in dirs:
            if want in d.name:
                picked.append(d)
                break
    if len(picked) < 3:
        picked = dirs[:3]
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for d in picked:
        frames = sorted(d.glob("*.jpg"))
        idxs = list(range(0, len(frames), max(1, len(frames) // 20)))[:20]
        for i in idxs:
            f = frames[i]
            (OUT / f"{d.name}_{f.name}").write_bytes(f.read_bytes())
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.iterdir()):
            z.write(f, f.name)
    print("frames:", len(list(OUT.iterdir())), "zip:", ZIP.stat().st_size)


def main():
    prepare_frames()

    r = run(["docker", "exec",
             "-e", f"DJANGO_SUPERUSER_USERNAME={SU}",
             "-e", f"DJANGO_SUPERUSER_PASSWORD={PW}",
             "-e", f"DJANGO_SUPERUSER_EMAIL={EMAIL}",
             "cvat_server", "bash", "-ic",
             "python3 ~/manage.py createsuperuser --noinput"])
    print("superuser rc:", r.returncode, r.stderr.strip()[-200:])

    rc, resp = curl_json("POST", f"{API}/auth/login",
                         {"username": SU, "password": PW})
    if rc != 0 or not resp or "key" not in resp:
        print("login failed:", resp)
        return
    tok = resp["key"]
    print("token ok")

    rc, projs = curl_json("GET", f"{API}/projects?search=标注SOP练习", auth=tok)
    pid = None
    if rc == 0 and projs and projs.get("results"):
        pid = projs["results"][0]["id"]
    if pid is None:
        rc, proj = curl_json("POST", f"{API}/projects",
                             {"name": "标注SOP练习-web-batch1"}, auth=tok)
        if rc != 0 or not proj or "id" not in proj:
            print("project failed:", proj)
            return
        pid = proj["id"]
    print("project:", pid)

    rc, lab = curl_json("PATCH", f"{API}/projects/{pid}",
                        {"labels": make_labels()}, auth=tok)
    if rc != 0:
        print("labels failed:", lab)
        return
    print("labels applied ok")

    rc, tasks = curl_json("GET", f"{API}/tasks?project_id={pid}", auth=tok)
    tid = None
    if rc == 0 and tasks and tasks.get("results"):
        for t in tasks["results"]:
            if t.get("name") == "深蹲60帧-标注练习":
                tid = t["id"]
    if tid is None:
        rc, task = curl_json("POST", f"{API}/tasks",
                             {"name": "深蹲60帧-标注练习", "project_id": pid,
                              "subset": "practice", "overlap": 0, "segment_size": 60},
                             auth=tok)
        if rc != 0 or not task or "id" not in task:
            print("task failed:", task)
            return
        tid = task["id"]
    print("task:", tid)

    r = run(["curl", "-s", "-m", "600", "-X", "POST", f"{API}/tasks/{tid}/data",
             "-H", f"Authorization: Token {tok}",
             "-F", f"client_files[0]=@{ZIP}", "-F", "image_quality=90"])
    print("upload rc:", r.returncode, r.stdout[:300])

    for _ in range(40):
        time.sleep(5)
        rc, t = curl_json("GET", f"{API}/tasks/{tid}", auth=tok)
        if t and t.get("status") in ("annotation", "completed", "validation"):
            print("task status:", t["status"], "size:", t.get("size"))
            return
    rc, t = curl_json("GET", f"{API}/tasks/{tid}", auth=tok)
    print("final:", t)


if __name__ == "__main__":
    main()
