"""
Serpentine Jewels — AI Photo Search
------------------------------------
Search your jewelry photo library by typing a description
OR uploading a reference photo. Powered by CLIP + Google Drive.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    Push this repo to GitHub, connect at share.streamlit.io
"""

import io
import os
import json
import base64
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Serpentine Jewels · Photo Search",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
INDEX_FILENAME   = "serpentine_photo_index.npz"
MODEL_NAME       = "ViT-B-32"
PRETRAINED       = "laion2b_s34b_b79k"
DRIVE_FOLDER_KEY = "drive_folder_id"   # stored in st.session_state / secrets
TOP_K_DEFAULT    = 20

# ── CSS — Serpentine brand palette ───────────────────────────────────────────
EMERALD = "#002F1E"
IVORY   = "#ECE8DF"
SAND    = "#E7E0D0"

st.markdown(f"""
<style>
  /* global background + font */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {IVORY};
      font-family: Arial, sans-serif;
  }}
  [data-testid="stSidebar"] {{
      background-color: {EMERALD} !important;
  }}
  [data-testid="stSidebar"] * {{ color: {IVORY} !important; }}
  /* header bar */
  .brand-header {{
      background: {EMERALD};
      color: {IVORY};
      padding: 1.4rem 2rem 1rem;
      border-radius: 8px;
      margin-bottom: 1.5rem;
  }}
  .brand-header h1 {{ margin: 0; font-family: Georgia, serif;
                      font-size: 2rem; color: {IVORY}; }}
  .brand-header p  {{ margin: 0.25rem 0 0; font-size: 0.95rem;
                      color: {SAND}; font-style: italic; }}
  /* result cards */
  .result-card {{
      background: white;
      border-radius: 8px;
      padding: 0.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      text-align: center;
  }}
  .result-card p {{
      font-size: 0.75rem;
      color: #555;
      margin: 0.3rem 0 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
  }}
  .score-badge {{
      display: inline-block;
      background: {EMERALD};
      color: {IVORY};
      font-size: 0.7rem;
      padding: 1px 6px;
      border-radius: 10px;
      margin-top: 4px;
  }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="brand-header">
  <h1>Serpentine Jewels · Photo Search</h1>
  <p>Type a description or upload a reference photo to find matching pieces</p>
</div>
""", unsafe_allow_html=True)

# ── Helpers: CLIP model ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model (first run only)…")
def load_clip():
    import torch
    import open_clip
    device = "mps" if hasattr(__import__("torch").backends, "mps") and \
              __import__("torch").backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, device


def embed_text(query: str):
    import torch
    model, _, tokenizer, device = load_clip()
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")[0]


def embed_image(pil_img: Image.Image):
    import torch
    model, preprocess, _, device = load_clip()
    with torch.no_grad():
        tensor = preprocess(pil_img).unsqueeze(0).to(device)
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")[0]


# ── Helpers: Google Drive ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Google Drive…")
def get_drive_service():
    """
    Authenticate with Google Drive.
    Credentials come from st.secrets["google_service_account"] (JSON key)
    when deployed on Streamlit Cloud, or from local credentials.json / token.json
    when running locally (same OAuth flow as the other scripts).
    """
    try:
        # ── Streamlit Cloud: service-account JSON in secrets ──
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        info = dict(st.secrets["google_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive"])
        return build("drive", "v3", credentials=creds)
    except (KeyError, FileNotFoundError):
        pass

    # ── Local: reuse OAuth token from the other scripts ──
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/drive"]
    TOKEN   = "token.json"
    CREDS   = "credentials.json"

    creds = None
    if os.path.exists(TOKEN):
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN, "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


def list_drive_images(service, folder_id: str) -> list[dict]:
    """Return all image files under a Drive folder (recursive)."""
    results, page_token, seen = [], None, set()

    def _recurse(fid):
        nonlocal page_token
        q = f"'{fid}' in parents and trashed = false"
        pt = None
        while True:
            resp = service.files().list(
                q=q,
                fields="nextPageToken, files(id, name, mimeType, thumbnailLink, webContentLink)",
                pageSize=1000, pageToken=pt,
            ).execute()
            for item in resp.get("files", []):
                if item["mimeType"] == "application/vnd.google-apps.folder":
                    if item["id"] not in seen:
                        seen.add(item["id"])
                        _recurse(item["id"])
                elif item["mimeType"].startswith("image/"):
                    results.append(item)
            pt = resp.get("nextPageToken")
            if not pt:
                break

    _recurse(folder_id)
    return results


def fetch_image_bytes(service, file_id: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


# ── Index management ──────────────────────────────────────────────────────────
def find_index_file(service, folder_id: str):
    """Return the Drive file ID of the index .npz if it exists."""
    resp = service.files().list(
        q=f"'{folder_id}' in parents and name='{INDEX_FILENAME}' and trashed=false",
        fields="files(id, name)",
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def load_index(service, folder_id: str):
    """Load the pre-built CLIP index from Drive into memory."""
    idx_id = find_index_file(service, folder_id)
    if not idx_id:
        return None
    raw = fetch_image_bytes(service, idx_id)
    data = np.load(io.BytesIO(raw), allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "ids":        [str(x) for x in data["ids"]],
        "names":      [str(x) for x in data["names"]],
    }


def save_index(service, folder_id: str, index: dict):
    """Save / overwrite the CLIP index .npz back to Drive."""
    from googleapiclient.http import MediaIoBaseUpload
    buf = io.BytesIO()
    np.savez(buf,
             embeddings=index["embeddings"],
             ids=np.array(index["ids"]),
             names=np.array(index["names"]))
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="application/octet-stream")

    existing_id = find_index_file(service, folder_id)
    if existing_id:
        service.files().update(fileId=existing_id, media_body=media).execute()
    else:
        meta = {"name": INDEX_FILENAME, "parents": [folder_id]}
        service.files().create(body=meta, media_body=media).execute()


def build_index(service, folder_id: str, progress_bar, existing_index=None):
    """
    Build or incrementally update the CLIP index.

    If existing_index is provided, only new photos (file IDs not already in
    the index) are processed. This makes daily updates take seconds rather
    than minutes -- only the genuinely new photos are touched.

    If existing_index is None, all photos are processed from scratch.
    """
    import torch
    model, preprocess, _, device = load_clip()

    all_files = list_drive_images(service, folder_id)

    # ── Incremental: skip anything already in the index ──────────────────────
    if existing_index is not None:
        indexed_ids = set(existing_index["ids"])
        # Also remove any IDs that no longer exist in Drive (deleted photos)
        current_ids = {f["id"] for f in all_files}
        still_valid = [
            (i, eid) for i, eid in enumerate(existing_index["ids"])
            if eid in current_ids
        ]
        if len(still_valid) < len(existing_index["ids"]):
            removed = len(existing_index["ids"]) - len(still_valid)
            st.info(f"Removed {removed} photo(s) from index that were deleted from Drive.")

        kept_idxs = [i for i, _ in still_valid]
        existing_embs  = existing_index["embeddings"][kept_idxs]
        existing_ids   = [existing_index["ids"][i]   for i in kept_idxs]
        existing_names = [existing_index["names"][i] for i in kept_idxs]
        indexed_ids    = set(existing_ids)

        new_files = [f for f in all_files if f["id"] not in indexed_ids]

        if not new_files:
            progress_bar.progress(1.0, text="Already up to date — no new photos found.")
            return existing_index  # nothing to do

        progress_bar.progress(0.0, text=f"Found {len(new_files)} new photo(s) to index…")
    else:
        new_files      = all_files
        existing_embs  = None
        existing_ids   = []
        existing_names = []

    # ── Embed only the new files ──────────────────────────────────────────────
    new_embeddings, new_ids, new_names = [], [], []
    for i, f in enumerate(new_files):
        progress_bar.progress(
            (i + 1) / len(new_files),
            text=f"Indexing {i+1}/{len(new_files)}: {f['name']}"
        )
        try:
            raw = fetch_image_bytes(service, f["id"])
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            with torch.no_grad():
                tensor = preprocess(pil).unsqueeze(0).to(device)
                feat = model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            new_embeddings.append(feat.cpu().numpy().astype("float32")[0])
            new_ids.append(f["id"])
            new_names.append(f["name"])
        except Exception as e:
            st.warning(f"Skipped {f['name']}: {e}")

    if not new_embeddings:
        # All new files errored out — return existing unchanged
        return existing_index

    new_emb_arr = np.stack(new_embeddings)

    # ── Merge new with existing ───────────────────────────────────────────────
    if existing_embs is not None and len(existing_embs) > 0:
        merged_embs  = np.concatenate([existing_embs, new_emb_arr], axis=0)
        merged_ids   = existing_ids   + new_ids
        merged_names = existing_names + new_names
    else:
        merged_embs  = new_emb_arr
        merged_ids   = new_ids
        merged_names = new_names

    return {
        "embeddings": merged_embs,
        "ids":        merged_ids,
        "names":      merged_names,
    }


def search(index: dict, query_vec: np.ndarray, top_k: int) -> list[tuple]:
    """Return list of (file_id, name, score) sorted by similarity."""
    sims = index["embeddings"] @ query_vec
    order = np.argsort(-sims)[:top_k]
    return [(index["ids"][i], index["names"][i], float(sims[i])) for i in order]


# ── Sidebar: config ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    folder_id = st.text_input(
        "Google Drive Folder ID",
        value=st.session_state.get("folder_id", ""),
        placeholder="18HQhjDRW3cLYlDJ…",
        help="The ID from the end of your Drive folder URL",
    )
    if folder_id:
        st.session_state["folder_id"] = folder_id

    top_k = st.slider("Results to show", 5, 50, TOP_K_DEFAULT, 5)

    st.markdown("---")
    st.markdown("### Index")
    st.caption(
        "The index lets search run instantly. **Update** adds only new photos "
        "(seconds). **Full Rebuild** reprocesses everything (minutes)."
    )
    update_btn  = st.button("Update (new photos only)", type="primary",
                             use_container_width=True)
    rebuild_btn = st.button("Full Rebuild", type="secondary",
                             use_container_width=True,
                             help="Reprocesses all photos from scratch. "
                                  "Use if search results seem wrong or stale.")

    st.markdown("---")
    st.markdown("### Search tips")
    st.caption("• **kite diamond three stone ring**")
    st.caption("• **vintage pearl necklace on white**")
    st.caption("• **gold bracelet lifestyle shot**")
    st.caption("• Upload a photo to find similar pieces")


# ── Main: connect & load index ────────────────────────────────────────────────
if not folder_id:
    st.info("Paste your Google Drive folder ID in the sidebar to get started.")
    st.stop()

try:
    service = get_drive_service()
except Exception as e:
    st.error(f"Could not connect to Google Drive: {e}")
    st.stop()

# ── Handle index build / update ───────────────────────────────────────────────
if update_btn or rebuild_btn:
    # For update: pass existing index so only new photos are processed.
    # For full rebuild: pass None so everything is reprocessed from scratch.
    existing = None
    if update_btn:
        # Load from session or Drive to use as baseline
        existing = st.session_state.get("index") or load_index(service, folder_id)
        label = "Checking for new photos…"
    else:
        label = "Rebuilding from scratch — this takes a few minutes…"

    with st.spinner(label):
        bar = st.progress(0)
        idx = build_index(service, folder_id, bar, existing_index=existing)
        save_index(service, folder_id, idx)
        st.session_state["index"] = idx

    added = len(idx["ids"]) - (len(existing["ids"]) if existing else 0)
    if update_btn and added == 0:
        st.success(f"Already up to date — {len(idx['ids'])} photos indexed.")
    elif update_btn:
        st.success(f"Added {added} new photo(s). Index now covers {len(idx['ids'])} photos.")
    else:
        st.success(f"Full rebuild complete — {len(idx['ids'])} photos indexed.")

# Load index (from session or Drive)
if "index" not in st.session_state:
    with st.spinner("Loading index from Drive…"):
        idx = load_index(service, folder_id)
        if idx:
            st.session_state["index"] = idx

index = st.session_state.get("index")
if index is None:
    st.warning(
        "No index found yet. Click **Build / Rebuild Index** in the sidebar "
        "to create one (takes a few minutes the first time)."
    )
    st.stop()

from datetime import datetime
_now = datetime.now().strftime("%b %d, %H:%M")
st.caption(f"Index contains **{len(index['ids'])} photos** · last loaded {_now}")

# ── Search UI ─────────────────────────────────────────────────────────────────
col_text, col_upload = st.columns([3, 1])
with col_text:
    query_text = st.text_input(
        "Search by description",
        placeholder="e.g. kite diamond engagement ring on white",
        label_visibility="collapsed",
    )
with col_upload:
    query_file = st.file_uploader(
        "Or upload a reference photo",
        type=["jpg", "jpeg", "png", "webp", "heic"],
        label_visibility="collapsed",
    )

# ── Run search ────────────────────────────────────────────────────────────────
results = []
search_label = ""

if query_file:
    pil_img = Image.open(query_file).convert("RGB")
    st.image(pil_img, caption="Reference photo", width=200)
    with st.spinner("Finding similar pieces…"):
        qvec = embed_image(pil_img)
    results = search(index, qvec, top_k)
    search_label = "Visually similar to your photo"

elif query_text.strip():
    with st.spinner("Searching…"):
        qvec = embed_text(query_text.strip())
    results = search(index, qvec, top_k)
    search_label = f"Results for **{query_text.strip()}**"

# ── Display results ───────────────────────────────────────────────────────────
if results:
    st.markdown(f"#### {search_label}")
    cols = st.columns(5)
    for rank, (fid, name, score) in enumerate(results):
        col = cols[rank % 5]
        with col:
            try:
                raw = fetch_image_bytes(service, fid)
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
                # Cap display size for speed
                pil.thumbnail((400, 400))
                st.image(pil, use_container_width=True)
            except Exception:
                st.markdown("_(preview unavailable)_")
            st.caption(f"{name[:28]}{'…' if len(name)>28 else ''}")
            st.caption(f"Match: {score:.0%}")

elif query_text.strip() or query_file:
    st.info("No results found — try different search terms.")
