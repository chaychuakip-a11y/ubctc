#!/usr/bin/env python3
"""
upload_gdrive.py — Upload a file to Google Drive using only Python stdlib.

Uses OAuth2 device authorization flow (no browser needed on server).
For large files, uses Google Drive resumable upload API.

Setup (one-time):
  1. Go to https://console.cloud.google.com/
  2. Create a project → Enable "Google Drive API"
  3. APIs & Services → Credentials → Create OAuth2 client ID → Desktop app
  4. Download JSON → save as client_secret.json in same dir as this script
     OR pass --client-id and --client-secret directly

Usage:
  python upload_gdrive.py <file_path> [--folder-id FOLDER_ID]
                          [--client-id CI --client-secret CS]
                          [--token-file token.json]
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB per chunk

# ─── OAuth2 device flow ──────────────────────────────────────────────────────

def load_client_credentials(args) -> tuple[str, str]:
    if args.client_id and args.client_secret:
        return args.client_id, args.client_secret
    secret_file = Path(args.client_secret_file)
    if not secret_file.exists():
        sys.exit(
            f"client_secret.json not found at {secret_file}.\n"
            "Download it from Google Cloud Console → APIs & Services → Credentials."
        )
    data = json.loads(secret_file.read_text())
    entry = data.get("installed") or data.get("web")
    return entry["client_id"], entry["client_secret"]


def device_auth(client_id: str, client_secret: str, token_file: Path) -> str:
    """Return a valid access token, refreshing or re-authorizing as needed."""
    if token_file.exists():
        tokens = json.loads(token_file.read_text())
        access_token = refresh_token(client_id, client_secret, tokens["refresh_token"])
        if access_token:
            tokens["access_token"] = access_token
            token_file.write_text(json.dumps(tokens, indent=2))
            return access_token

    # Step 1: request device code
    resp = post_json(
        "https://oauth2.googleapis.com/device/code",
        {
            "client_id": client_id,
            "scope": "https://www.googleapis.com/auth/drive.file",
        },
    )
    print(f"\nOpen this URL in your browser:\n  {resp['verification_url']}")
    print(f"Enter code: {resp['user_code']}\n")
    print("Waiting for authorization", end="", flush=True)

    interval = resp.get("interval", 5)
    expires_in = resp.get("expires_in", 1800)
    deadline = time.time() + expires_in

    while time.time() < deadline:
        time.sleep(interval)
        print(".", end="", flush=True)
        token = poll_token(client_id, client_secret, resp["device_code"])
        if token:
            print(" authorized!\n")
            token_file.write_text(json.dumps(token, indent=2))
            return token["access_token"]

    sys.exit("\nAuthorization timed out.")


def refresh_token(client_id: str, client_secret: str, refresh_tok: str):
    try:
        resp = post_json(
            "https://oauth2.googleapis.com/token",
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_tok,
                "grant_type": "refresh_token",
            },
        )
        return resp.get("access_token")
    except Exception:
        return None


def poll_token(client_id, client_secret, device_code):
    try:
        resp = post_json(
            "https://oauth2.googleapis.com/token",
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        if "access_token" in resp:
            return resp
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        if body.get("error") not in ("authorization_pending", "slow_down"):
            raise
    return None


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def post_json(url: str, data: dict) -> dict:
    payload = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def api_request(method: str, url: str, token: str,
                headers: dict = None, data: bytes = None) -> dict:
    h = {"Authorization": f"Bearer {token}"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req) as r:
        body = r.read()
        return json.loads(body) if body else {}


# ─── Google Drive upload ──────────────────────────────────────────────────────

def create_folder(token: str, name: str, parent_id: str = None) -> str:
    """Create a folder and return its ID."""
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    data = json.dumps(meta).encode()
    resp = api_request(
        "POST",
        "https://www.googleapis.com/drive/v3/files",
        token,
        headers={"Content-Type": "application/json"},
        data=data,
    )
    return resp["id"]


def find_or_create_folder(token: str, name: str, parent_id: str = None) -> str:
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode({
        "q": q, "fields": "files(id,name)", "pageSize": "1"
    })
    resp = api_request("GET", url, token)
    files = resp.get("files", [])
    if files:
        print(f"Found existing folder '{name}' (id={files[0]['id']})")
        return files[0]["id"]
    fid = create_folder(token, name, parent_id)
    print(f"Created folder '{name}' (id={fid})")
    return fid


def resumable_upload(token: str, file_path: Path, folder_id: str = None) -> str:
    """Upload large file using resumable upload. Returns file ID."""
    file_size = file_path.stat().st_size
    print(f"Uploading: {file_path.name} ({file_size/1024**3:.2f} GB)")

    meta = {"name": file_path.name}
    if folder_id:
        meta["parents"] = [folder_id]

    # Initiate resumable session
    init_url = (
        "https://www.googleapis.com/upload/drive/v3/files"
        "?uploadType=resumable"
    )
    data = json.dumps(meta).encode()
    req = urllib.request.Request(
        init_url, data=data, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Upload-Content-Type": "application/gzip",
            "X-Upload-Content-Length": str(file_size),
        },
    )
    with urllib.request.urlopen(req) as r:
        session_url = r.headers["Location"]

    # Upload chunks
    uploaded = 0
    start_time = time.time()
    with open(file_path, "rb") as f:
        while uploaded < file_size:
            chunk = f.read(CHUNK_SIZE)
            end = uploaded + len(chunk) - 1
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {uploaded}-{end}/{file_size}",
            }
            req = urllib.request.Request(session_url, data=chunk,
                                         headers=headers, method="PUT")
            try:
                with urllib.request.urlopen(req) as r:
                    if r.status in (200, 201):
                        result = json.loads(r.read())
                        elapsed = time.time() - start_time
                        speed = file_size / elapsed / 1024**2
                        print(f"  Upload complete in {elapsed:.0f}s ({speed:.1f} MB/s)")
                        return result["id"]
            except urllib.error.HTTPError as e:
                if e.code == 308:  # Resume Incomplete — chunk accepted
                    pass
                else:
                    raise
            uploaded += len(chunk)
            pct = uploaded / file_size * 100
            speed = uploaded / (time.time() - start_time) / 1024**2
            print(f"  {pct:.1f}%  {uploaded/1024**2:.0f}/{file_size/1024**2:.0f} MB  "
                  f"{speed:.1f} MB/s", flush=True)

    return ""  # should not reach here


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="File to upload")
    ap.add_argument("--folder-id", default=None,
                    help="Google Drive folder ID to upload into (optional)")
    ap.add_argument("--folder-name", default="tts_kr_num",
                    help="Create/use a subfolder with this name (default: tts_kr_num)")
    ap.add_argument("--client-id", default=None)
    ap.add_argument("--client-secret", default=None)
    ap.add_argument("--client-secret-file", default="client_secret.json",
                    help="Path to OAuth2 client secret JSON (default: client_secret.json)")
    ap.add_argument("--token-file", default=str(Path.home() / ".gdrive_token.json"),
                    help="Where to cache OAuth tokens")
    args = ap.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        sys.exit(f"File not found: {file_path}")

    client_id, client_secret = load_client_credentials(args)
    token_file = Path(args.token_file)
    access_token = device_auth(client_id, client_secret, token_file)

    # Resolve target folder
    folder_id = args.folder_id
    if args.folder_name:
        folder_id = find_or_create_folder(access_token, args.folder_name, folder_id)

    file_id = resumable_upload(access_token, file_path, folder_id)
    print(f"\nDone! Google Drive file ID: {file_id}")
    print(f"View: https://drive.google.com/file/d/{file_id}/view")


if __name__ == "__main__":
    main()
