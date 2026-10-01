import json
import subprocess
import time
import zipfile
from pathlib import Path

API = "http://127.0.0.1:8080/api"
SU, PW = "ethan", "isef2027-anno"
BASE = Path("/home/ubuntu/isef/data/raw/web-batch1")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def curl_raw(args):
    r = run(["curl", "-s", "-m", "300", "-w", "\n%{http_code}"] + args)
    body, _, code = r.stdout.rpartition("\n")
    return code.strip(), body[:400]


def login():
    r = run(["curl", "-s", "-m", "60", "-X", "POST", f"{API}/auth/login",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"username": SU, "password": PW})])
    return json.loads(r.stdout)["key"]


def make_tiny_zip():
    zpath = BASE / "tiny-test.zip"
    frames = sorted((BASE / "practice-annot").glob("*.jpg"))[:2]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for i, f in enumerate(frames):
            z.write(f, f"test_{i}.jpg")
    return zpath


tok = login()
print("token ok")

zipf = make_tiny_zip()

code, body = curl_raw(["-X", "POST", f"{API}/tasks/1/data",
                       "-H", f"Authorization: Token {tok}",
                       "-H", "Upload-Start: *",
                       "-H", "Upload-Finish: *",
                       "-F", f"client_files=@{zipf}",
                       "-F", "image_quality=90"])
print("tus-headers upload:", code, body)

time.sleep(8)
r = run(["docker", "exec", "cvat_db", "psql", "-U", "root", "-d", "cvat",
         "-c", "SELECT id, file, data_id FROM engine_clientfile ORDER BY id DESC LIMIT 5;"])
print("clientfile rows:\n", r.stdout)

r = run(["docker", "run", "--rm", "-v", "cvat_cvat_data:/data", "busybox",
         "find", "/data/data/1", "-maxdepth", "3"])
print("data dir:\n", r.stdout)
