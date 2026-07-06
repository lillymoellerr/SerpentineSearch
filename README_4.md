# Serpentine Jewels — AI Photo Search

Search your jewelry photo library by typing a description ("kite diamond
three stone ring") or uploading a reference photo. No manual tags needed.

## How it works

1. The app builds an AI "index" of your Google Drive photo folder — a small
   file (~5-10MB) that captures the visual essence of every photo. This is
   saved back to your Drive folder so it persists.
2. When you search, your text or reference photo is converted to the same
   kind of visual fingerprint and compared against the index.
3. The most visually similar photos are returned, ranked by match score.

No photos are ever stored locally — everything reads directly from Drive.

---

## Running locally

### Prerequisites
- Python 3.10+
- The `credentials.json` + `token.json` from the Google Cloud OAuth setup
  you already did for the other scripts (copy them into this folder)

### Install & run
```bash
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens at http://localhost:8501

1. Paste your Drive folder ID in the sidebar (the Winners folder:
   `18HQhjDRW3cLYlDJxktK50HqObl4GvmIz` or whichever you want to search)
2. Click **Build / Rebuild Index** — takes ~5-10 mins for 1,210 photos,
   saves the result to Drive so future loads are instant
3. Search by typing or uploading a photo

---

## Deploying to Streamlit Cloud (shareable link for the whole team)

This is the "anyone on the team opens a link" version. Free tier works.

### Step 1: Push to GitHub
Create a private GitHub repo and push these three files:
- `app.py`
- `requirements.txt`
- `README.md`

Do NOT push `credentials.json` or `token.json` — those stay local.

### Step 2: Create a Google Service Account
The deployed app can't do the browser OAuth flow, so it needs a service
account (a non-human Google identity that can access Drive).

1. Go to console.cloud.google.com → your "Serpentine Photo Tool" project
2. IAM & Admin → Service Accounts → Create Service Account
   - Name: "serpentine-search-app"
   - Role: "Editor" (or "Drive File Viewer" for read-only)
3. Click the service account → Keys → Add Key → Create new key → JSON
4. Download the JSON file — keep it private

5. Share your Drive folder with the service account's email address
   (it looks like serpentine-search-app@...iam.gserviceaccount.com)
   — go to the Drive folder → Share → paste that email → Editor

### Step 3: Add the key to Streamlit secrets
1. Go to share.streamlit.io → New app → connect your GitHub repo
2. In Advanced settings → Secrets, paste the service account JSON like this:

```toml
[google_service_account]
type = "service_account"
project_id = "serpentine-photo-tool"
private_key_id = "..."
private_key = "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
client_email = "serpentine-search-app@....iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

3. Deploy — Streamlit gives you a URL like
   `https://serpentine-jewels-search.streamlit.app`
   Share that link with the team.

### Access control
Free Streamlit Cloud apps are public by default. To restrict to your team:
- Streamlit Teams plan ($20/month) adds login-gating by Google account
- OR share the URL only internally and rely on the Drive folder ID being
  non-obvious (not a security solution, but fine for a small internal team)

---

## Rebuilding the index

Run **Build / Rebuild Index** in the sidebar whenever:
- New photos are added to the Drive folder
- You want to switch to a different folder

The index file (`serpentine_photo_index.npz`) is saved directly to your
Drive folder and loaded automatically on the next session.

---

## Search tips

- Be specific: **"kite diamond three stone platinum"** beats **"ring"**
- Shot type works: **"on white"**, **"lifestyle"**, **"on model"**
- Metal + stone: **"yellow gold pave bracelet"**
- Upload a photo of a piece you like to find similar ones
- Match scores above ~65% are strong visual matches
