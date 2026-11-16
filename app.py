import hmac
import io
import os
import tarfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone

import chromadb
import requests
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---------- Config ----------
# Model names change over time. If you get a "model not found" error, check
# https://ai.google.dev/gemini-api/docs/models and edit these.
# Fallback list, used only if the live model list cannot be fetched
CHAT_MODELS = ["gemini-3.8-flash"]
EMBED_MODEL = "gemini-embedding-001"

# Agent mode (Antigravity managed agent). Preview agent names change; check
# https://ai.google.dev/gemini-api/docs/managed-agents-quickstart
AGENT_ID = "antigravity-preview-09-2026"
AGENT_DAILY_LIMIT = 20  # app-wide cap per day, to protect your free quota

PERSONAS = {
    "Helpful assistant": "You are a helpful, concise assistant.",
    "Study buddy": "You are a friendly study buddy. Explain concepts simply, "
                   "use examples, and quiz the user when helpful.",
    "Coding tutor": "You are a patient coding tutor. Explain step by step, "
                    "show short code examples, and point out common mistakes.",
    "Custom": "",
}

st.set_page_config(page_title="My AI Chatbot", page_icon="🤖")
st.title("🤖 My AI Chatbot")

# ---------- API key (Streamlit secrets or environment variable) ----------
api_key = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Add GEMINI_API_KEY to your Streamlit secrets (or environment).")
    st.stop()


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


@st.cache_resource
def get_client(key):
    return genai.Client(api_key=key)


client = get_client(api_key)


@st.cache_data(ttl=3600)
def list_chat_models(_client):
    """Ask the API which Gemini chat models this key can use right now."""
    try:
        skip = ("image", "tts", "live", "audio", "embedding", "robotics", "computer")
        names = []
        for m in _client.models.list():
            n = m.name.replace("models/", "")
            if (
                "generateContent" in (m.supported_actions or [])
                and n.startswith("gemini")
                and "flash" in n
                and not any(x in n for x in skip)
            ):
                names.append(n)
        return sorted(set(names)) or CHAT_MODELS
    except Exception:
        return CHAT_MODELS

# ---------- Session state ----------
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "indexed" not in st.session_state:
    st.session_state.indexed = set()
for _k, _v in {
    "agent_ok": False,     # password entered correctly
    "agent_prev": None,    # id of the last agent interaction (for follow-ups)
    "agent_env": None,     # id of the agent's sandbox environment
    "agent_files": None,   # files downloaded from the sandbox
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if "chroma" not in st.session_state:
    st.session_state.chroma = chromadb.Client()
    st.session_state.collection = st.session_state.chroma.create_collection(
        f"docs_{uuid.uuid4().hex}"
    )


# ---------- Helpers ----------
def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


def embed(texts):
    res = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in res.embeddings]


def index_pdf(file):
    reader = PdfReader(file)
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    chunks = chunk_text(text)
    col = st.session_state.collection
    for i in range(0, len(chunks), 32):
        batch = chunks[i:i + 32]
        col.add(
            ids=[f"{file.name}-{i + j}" for j in range(len(batch))],
            documents=batch,
            embeddings=embed(batch),
            metadatas=[{"source": file.name}] * len(batch),
        )
    return len(chunks)


def retrieve(question, k=4):
    col = st.session_state.collection
    if col.count() == 0:
        return [], []
    res = col.query(query_embeddings=embed([question]), n_results=k)
    return res["documents"][0], res["metadatas"][0]


# ---------- Agent mode helpers ----------
@st.cache_resource
def _agent_state():
    return {"day": None, "n": 0, "lock": threading.Lock()}


def _today_pacific():
    # Google's daily quota resets at midnight Pacific time
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def agent_runs_today():
    state = _agent_state()
    with state["lock"]:
        if state["day"] != _today_pacific():
            state["day"], state["n"] = _today_pacific(), 0
        return state["n"]


def agent_try_use():
    """Count one agent run. Returns False if today's cap is reached."""
    state = _agent_state()
    with state["lock"]:
        if state["day"] != _today_pacific():
            state["day"], state["n"] = _today_pacific(), 0
        if state["n"] >= AGENT_DAILY_LIMIT:
            return False
        state["n"] += 1
        return True


def run_agent(text, instruction):
    if not hasattr(client, "interactions"):
        raise RuntimeError(
            "This google-genai version has no Interactions API. "
            "Set google-genai>=2.3.0 in requirements.txt."
        )
    kwargs = {"agent": AGENT_ID, "input": text}
    if st.session_state.agent_prev and st.session_state.agent_env:
        # follow-up: continue the same conversation and sandbox
        kwargs["previous_interaction_id"] = st.session_state.agent_prev
        kwargs["environment"] = st.session_state.agent_env
    else:
        kwargs["environment"] = "remote"  # fresh sandbox
        if instruction:
            kwargs["system_instruction"] = instruction
    interaction = client.interactions.create(**kwargs)
    st.session_state.agent_prev = interaction.id
    st.session_state.agent_env = interaction.environment_id
    return interaction


def show_steps(interaction):
    steps = getattr(interaction, "steps", None) or []
    if steps:
        with st.expander(f"Agent steps ({len(steps)})"):
            for i, step in enumerate(steps, 1):
                kind = getattr(step, "type", None) or type(step).__name__
                st.caption(f"{i}. {kind}")


def fetch_workspace(env_id):
    r = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/files/environment-{env_id}:download",
        params={"alt": "media"},
        headers={"x-goog-api-key": api_key},
        timeout=60,
    )
    r.raise_for_status()
    return r.content


def archive_kind(data):
    if data[:2] == b"PK":
        return ".zip", "application/zip"
    if data[:2] == b"\x1f\x8b":
        return ".tar.gz", "application/gzip"
    if data[257:262] == b"ustar":
        return ".tar", "application/x-tar"
    return ".bin", "application/octet-stream"


def extract_images(data):
    exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    imgs = []
    try:
        if data[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    if name.lower().endswith(exts) and z.getinfo(name).file_size < 5_000_000:
                        imgs.append((name, z.read(name)))
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
                for mem in t.getmembers():
                    if mem.isfile() and mem.name.lower().endswith(exts) and mem.size < 5_000_000:
                        imgs.append((mem.name, t.extractfile(mem).read()))
    except Exception:
        pass
    return imgs[:8]


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    available = list_chat_models(client)
    # Lite models have higher free-tier limits, so make the newest one the default
    lite = [m for m in available if "lite" in m]
    default = available.index(lite[-1]) if lite else 0
    model = st.selectbox("Model", available, index=default)
    persona = st.selectbox("Personality", list(PERSONAS))
    if persona == "Custom":
        system_prompt = st.text_area(
            "System prompt", "You are a pirate who explains things clearly."
        )
    else:
        system_prompt = PERSONAS[persona]
    temperature = st.slider("Creativity", 0.0, 2.0, 0.7, 0.1)

    st.divider()
    st.header("Image chat")
    image_file = st.file_uploader(
        "Attach an image", type=["png", "jpg", "jpeg", "webp"]
    )
    if image_file:
        st.image(image_file)
        st.caption("Attached to your messages until you remove it.")

    st.divider()
    st.header("Document Q&A")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    for f in files or []:
        if f.name not in st.session_state.indexed:
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    n = index_pdf(f)
                    st.session_state.indexed.add(f.name)
                    st.success(f"{f.name}: {n} chunks")
                except Exception as e:
                    st.error(f"Could not index {f.name}: {e}")
    if st.session_state.indexed:
        st.caption("Indexed: " + ", ".join(sorted(st.session_state.indexed)))

    st.divider()
    st.header("Agent mode")
    agent_mode = st.toggle(
        "Use Antigravity agent",
        help="Runs code in a Google sandbox. Password protected to save quota.",
    )
    agent_active = False
    if agent_mode:
        agent_pw = get_secret("AGENT_PASSWORD")
        if not agent_pw:
            st.warning("Agent mode is locked. Add AGENT_PASSWORD to the app secrets.")
        elif st.session_state.agent_ok:
            agent_active = True
        else:
            entered = st.text_input("Agent password", type="password")
            if entered and hmac.compare_digest(entered.encode(), str(agent_pw).encode()):
                st.session_state.agent_ok = True
                st.rerun()
            elif entered:
                st.error("Wrong password.")
    if agent_active:
        st.caption(f"Agent runs today: {agent_runs_today()} / {AGENT_DAILY_LIMIT}")
        st.caption("Images and PDFs are ignored in agent mode.")
        if st.session_state.agent_env and st.button("Load agent files"):
            try:
                with st.spinner("Fetching files from the agent workspace..."):
                    data = fetch_workspace(st.session_state.agent_env)
                st.session_state.agent_files = {
                    "data": data,
                    "images": extract_images(data),
                }
            except Exception as e:
                st.error(f"Could not fetch files: {e}")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.msgs = []
        st.session_state.agent_prev = None
        st.session_state.agent_env = None
        st.session_state.agent_files = None
        st.rerun()
    if st.button("Clear documents"):
        st.session_state.chroma.delete_collection(st.session_state.collection.name)
        st.session_state.collection = st.session_state.chroma.create_collection(
            f"docs_{uuid.uuid4().hex}"
        )
        st.session_state.indexed = set()
        st.rerun()

# ---------- Chat history ----------
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        if m.get("image"):
            st.image(m["image"], width=250)
        st.write(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.caption(s)

if st.session_state.agent_files:
    with st.expander("📁 Agent files", expanded=True):
        files_data = st.session_state.agent_files["data"]
        for name, img in st.session_state.agent_files["images"]:
            st.image(img, caption=name)
        ext, mime = archive_kind(files_data)
        st.download_button(
            "Download all agent files",
            files_data,
            file_name=f"agent_workspace{ext}",
            mime=mime,
        )

# ---------- New message ----------
if prompt := st.chat_input("Ask me anything (or about your PDFs)"):
    image_bytes = image_file.getvalue() if image_file else None
    image_mime = image_file.type if image_file else None
    st.session_state.msgs.append(
        {"role": "user", "content": prompt, "image": image_bytes}
    )
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=250)
        st.write(prompt)

    if agent_active:
        with st.chat_message("assistant"):
            if image_bytes:
                st.info("Agent mode ignores attached images.")
            if not agent_try_use():
                reply = (
                    f"Agent mode reached today's limit of {AGENT_DAILY_LIMIT} runs. "
                    "Turn it off to keep chatting normally, or try again tomorrow."
                )
                st.warning(reply)
            else:
                try:
                    with st.spinner("Agent is working... this can take a minute or two."):
                        interaction = run_agent(prompt, system_prompt)
                    reply = interaction.output_text or "(The agent returned no text.)"
                    st.write(reply)
                    show_steps(interaction)
                    st.caption("If the agent made files, click Load agent files in the sidebar.")
                except Exception as e:
                    reply = (
                        f"Agent error: {e}\n\n503 means Google is busy, 429 means a "
                        "quota limit. Wait a bit and try again."
                    )
                    st.error(reply)
        st.session_state.msgs.append(
            {"role": "assistant", "content": reply, "sources": []}
        )
        st.stop()

    docs, metas = [], []
    if not image_bytes:  # document search is skipped when an image is attached
        try:
            docs, metas = retrieve(prompt)
        except Exception as e:
            st.warning(f"Document search failed: {e}")

    user_text = prompt
    if docs:
        context = "\n\n---\n\n".join(docs)
        user_text = (
            "Answer using the document excerpts below. If the answer is not "
            "in them, say so, then answer from general knowledge.\n\n"
            f"Excerpts:\n{context}\n\nQuestion: {prompt}"
        )

    # Gemini uses the roles "user" and "model"; history stays text-only
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in st.session_state.msgs[:-1]
    ]
    last_parts = [types.Part.from_text(text=user_text)]
    if image_bytes:
        last_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
    contents.append(types.Content(role="user", parts=last_parts))

    def is_busy(e):
        return "503" in str(e) or "UNAVAILABLE" in str(e)

    def is_gone(e):
        return "404" in str(e) or "NOT_FOUND" in str(e)

    def is_limit(e):
        return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)

    def stream():
        # Try the chosen model first. If Google says it is busy (503), retry
        # once, then fall back to the next available model automatically.
        candidates = ([model] + [m for m in reversed(available) if m != model])[:4]
        last_err = None
        for cand in candidates:
            for attempt in range(2):
                started = False
                try:
                    for chunk in client.models.generate_content_stream(
                        model=cand,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                        ),
                    ):
                        if chunk.text:
                            started = True
                            yield chunk.text
                    return
                except Exception as e:
                    if started or not (is_busy(e) or is_gone(e) or is_limit(e)):
                        raise
                    last_err = e
                    if is_gone(e) or is_limit(e):
                        break  # unusable or out of quota: try the next model
                    time.sleep(2)  # busy: wait briefly, then retry
        raise last_err

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream())
        except Exception as e:
            reply = (
                f"Something went wrong: {e}\n\n503 means Google is busy, so try "
                "again in a minute. 429 or quota means you've hit the "
                "free-tier limit, so wait a bit."
            )
            st.error(reply)

        sources = sorted({m["source"] for m in metas}) if docs else []
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.caption(s)

    st.session_state.msgs.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
