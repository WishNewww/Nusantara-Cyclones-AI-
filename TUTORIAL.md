# Nusantara Watch AI Backend — Google Colab Tutorial

This revises the previous local-machine tutorial for running entirely
inside **Google Colab**. The backend code itself (`build_vector.py`,
`rag.py`) doesn't change — what changes is *how you get the code and
data onto the runtime, how you store secrets, and what to expect from
Colab's ephemeral VM*, which is the part that actually trips people up.

Each `▶` block below is one Colab **cell** — paste each into its own
cell, run top to bottom on a fresh runtime.

---

## 0. What's different about Colab (read this first)

- **The VM is ephemeral.** Every fresh runtime starts with an empty
  `/content` — no code, no data, no installed packages, no cached
  embedding model. Everything in §2–§5 below has to be redone whenever
  you connect to a new runtime (e.g. after `Runtime → Disconnect and
  delete runtime`, or after a long idle timeout).
- **Free-tier CPU is enough.** `all-MiniLM-L6-v2` (the local embedding
  model) is small — you do not need to switch to a GPU runtime for this
  backend. GPU only matters if you later swap in a much larger embedding
  or generation model.
- **One Colab session = one process**, the whole time you keep that
  runtime open. This is actually a good fit for how `rag.py` is built:
  `_get_backend_resources()` caches the embedding model and FAISS index
  once per process and reuses them across every cell/question after
  that — in a Streamlit deployment that caching had to be designed
  carefully around *multiple* concurrent users sharing one process (see
  the earlier debugging notes if you have that version); in a single
  Colab notebook there's only ever one user, so it just works.

---

## 1. Open a fresh notebook and confirm the runtime

```python
# ▶ Cell 1 — sanity check
!python --version
!nvidia-smi 2>/dev/null || echo "No GPU attached — that's fine for this backend."
```

You don't need to change the runtime type. `Runtime → Change runtime
type → CPU` is sufficient.

---

## 2. Get the code onto the runtime

Pick **one** of these two — whichever matches where your code actually
lives right now.

**Option A — clone from GitHub** (once you've pushed this project there,
which is the intended end state for the portfolio):

```python
# ▶ Cell 2A
!git clone https://github.com/<your-username>/NusantaraWatchAI.git
%cd NusantaraWatchAI
```

**Option B — upload the files directly** (works right now, before
anything is pushed to GitHub):

```python
# ▶ Cell 2B
import os
os.makedirs("NusantaraWatchAI/data", exist_ok=True)
%cd NusantaraWatchAI

from google.colab import files
print("Select build_vector.py, rag.py, requirements.txt, inspect_retrieval.py:")
uploaded = files.upload()
```

This opens a file picker in the Colab UI — select the four backend
files (not the CSVs yet, those go in step 4).

---

## 3. Install dependencies

```python
# ▶ Cell 3
!pip install -q -r requirements.txt
```

Takes roughly 1-2 minutes on Colab — `sentence-transformers` and
`faiss-cpu` are the largest installs.

> **If you see a dependency-resolution error or an import fails right
> after this cell**, it's almost always Colab's *pre-installed* package
> versions (Colab ships with its own `numpy`/`grpcio`/`protobuf`
> already loaded into the running process) conflicting with a freshly
> installed one. Fix: `Runtime → Restart session` (not "disconnect and
> delete" — that also wipes your files), then re-run cells 2 onward.
> You do not need to reinstall from scratch, just restart the Python
> process.

---

## 4. Set your Gemini API key

Colab's **Secrets** manager (the 🔑 icon in the left sidebar) is the
right way to do this — it keeps the key out of the notebook file itself
(important if you ever share or publish the `.ipynb`).

1. Click the 🔑 icon in the left sidebar.
2. Add a new secret named `GEMINI_API_KEY`, paste your key as the
   value, and toggle **Notebook access** on for this notebook.

```python
# ▶ Cell 4
import os
from google.colab import userdata

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
print("Key loaded:", bool(os.environ.get("GEMINI_API_KEY")))
```

If you're on classic Colab without Secrets access, fall back to a
masked prompt instead — never hardcode the key in a cell:

```python
# ▶ Cell 4 (fallback)
import os
from getpass import getpass

os.environ["GEMINI_API_KEY"] = getpass("Enter your Gemini API key: ")
```

Either way, `rag.py`'s `load_dotenv()` is a no-op here (no `.env` file
exists), which is fine — `os.environ` is already set directly, and both
`GOOGLE_API_KEY` and `GEMINI_API_KEY` are read by the underlying client.

---

## 5. Get the dataset onto the runtime

Same two options as step 2 — pick whichever is less friction for you:

**Option A — upload directly (simplest for a one-off session):**

```python
# ▶ Cell 5A
from google.colab import files
print("Select western_pacific.csv and southeast_indian.csv:")
uploaded = files.upload()

import shutil
for fname in uploaded:
    shutil.move(fname, f"data/{fname}")
!ls data/
```

**Option B — mount Google Drive (recommended if you'll run this more
than once):** this is the one genuinely Colab-specific decision worth
making deliberately, since uploading the same two CSVs by hand every
session gets old fast, and — more importantly — it lets you persist the
*built FAISS index* too, so you skip re-embedding 2,303 documents on
every fresh runtime.

```python
# ▶ Cell 5B
from google.colab import drive
drive.mount("/content/drive")

# One-time: put your CSVs in this Drive folder ahead of time via the
# Drive web UI, then symlink them in so build_vector.py finds them at
# the normal relative path.
DRIVE_DATA = "/content/drive/MyDrive/NusantaraWatchAI/data"
!ln -sf {DRIVE_DATA} data
!ls data/
```

Either way, confirm both CSVs are present before continuing:

```python
# ▶ Cell 5 (verify)
import os
expected = {"western_pacific.csv", "southeast_indian.csv"}
found = set(os.listdir("data"))
missing = expected - found
print("Missing:", missing or "none — good to go")
```

---

## 6. Build the vector index

```python
# ▶ Cell 6
!python build_vector.py
```

Expect:

```
INFO | Loaded 108320 observations across 2 region file(s).
INFO | Built 2303 cyclone document(s) from 108320 observation(s).
INFO | Embedding 2303 document(s) with HuggingFace (sentence-transformers/all-MiniLM-L6-v2)...
INFO | FAISS index persisted to '.../faiss_index'.
```

First run downloads the MiniLM weights from the Hugging Face Hub
(~90MB) — Colab's network is fast, this is typically well under a
minute, then it's cached for the rest of the session.

**If you mounted Drive in step 5B**, persist the index there too so
next session skips this step entirely:

```python
# ▶ Cell 6 (optional, only if using Drive)
!cp -r faiss_index /content/drive/MyDrive/NusantaraWatchAI/faiss_index_backup
```

Next session, restore instead of rebuilding: `!cp -r
/content/drive/MyDrive/NusantaraWatchAI/faiss_index_backup faiss_index`.
Only rebuild for real if you've changed the CSVs or the document-
building code — a restored index and freshly-changed code will silently
drift apart otherwise.

---

## 7. Ask questions — the notebook-native way

You could shell out with `!python rag.py "..."` like the CLI tutorial
did, but in a notebook it's more natural (and much faster across
repeated questions, since it reuses the cached model/index instead of
restarting a fresh Python process each time) to import and call
`ask_question` directly:

```python
# ▶ Cell 7
from rag import ask_question

print(ask_question("Tell me about Cyclone Seroja."))
```

```python
# ▶ Cell 8 — subsequent questions reuse the already-loaded model/index
print(ask_question("Compare Cyclone Seroja and Cyclone Tracy."))
print(ask_question("Which cyclones occurred in Western Pacific during 1998?"))
```

The first call in a session is slower (loading the embedding model
into memory); every call after that in the same runtime is fast.

---

## 8. Validate retrieval without spending an LLM call

Same diagnostic tool as before, called the notebook-native way instead
of via `!python`:

```python
# ▶ Cell 9
from inspect_retrieval import show_index_stats, inspect_query

show_index_stats()
```

```python
# ▶ Cell 10
inspect_query("Tell me about Cyclone ANN 1.")
```

This prints the top-5 retrieved cyclones with FAISS distance scores,
bypassing Gemini entirely — if the cyclone you expect isn't in that
list, it's a retrieval/embedding issue, not a prompt or LLM issue. Worth
running this once after every fresh index build as a smoke test, not
just when something looks wrong.

---

## 9. Why the embedding text and LLM context are different

(Unchanged from the local tutorial, and it matters just as much on
Colab.) `all-MiniLM-L6-v2` truncates at 256 tokens and mean-pools
whatever survives into one vector. On this dataset, 93.8% of the 2,303
cyclone documents exceeded 256 words, and the tokens that *did* survive
were dominated by a `Key : value` template identical across every
document — so the few tokens actually distinguishing one cyclone from
another got statistically swamped, and retrieval degraded toward
near-random for direct name queries (e.g. "Cyclone ANN 1" failing to
retrieve the ANN 1 document at all, despite it existing correctly in
the index).

The fix already built into `build_vector.py`: each cyclone gets a
short (~60-90 word) natural-language summary, name front-loaded, used
*only* for embedding; the full detailed report is kept in
`metadata["full_text"]` and is what actually reaches Gemini after
retrieval. If you ever swap the embedding model, re-run `--stats` /
`inspect_query` before assuming a quality change is real — check
whether it's a genuine regression or this same truncation pattern
recurring under a different context window.

---

## 10. Known limitations (by design, not bugs)

- **Aggregate/filter queries** ("which cyclones occurred in Western
  Pacific during 1998?") are answered from whichever 5 documents are
  semantically closest — not a guaranteed-complete scan. No SQL/agent
  layer exists here by explicit scope, so exhaustive aggregation isn't
  guaranteed; try `ask_question(question, max_docs=15)` if you want a
  wider (still not exhaustive) net for this class of question.
- **Only Gemini is implemented** for generation. `model=` is accepted
  for forward-compatibility with a future frontend, but anything other
  than `"Gemini"` returns a clear "not supported" message.
- **Missing `INTENSITY` values** are common (~18% of rows); profile
  stats degrade to `"Data unavailable"` rather than crashing or
  fabricating a number.

---

## 11. Colab-specific troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` right after `pip install` | Colab's package cache from a previous session, or the install cell was skipped after a runtime restart | `Runtime → Restart session`, then re-run cells 2→3 in order |
| `FileNotFoundError: No FAISS index found` | `build_vector.py` (step 6) was never run this session, or you `%cd`'d somewhere else first | Confirm `!pwd` shows the project folder, then re-run step 6 |
| API key error even after step 4 | Secret's "Notebook access" toggle wasn't enabled, or you're on a different notebook than the one you granted access to | Re-check the 🔑 panel; fall back to the `getpass` cell if Secrets access is being unreliable |
| Everything worked yesterday, now nothing is found | Fresh runtime = empty `/content`; nothing persists without Drive | Expected behavior — either redo steps 2-6, or set up Drive persistence (step 5B / step 6 optional cell) |
| Cell 6 seems to hang | First-run model download over a slow connection, or a very large `data/` upload still finishing | Check the Colab "RAM/Disk" usage graph top-right; genuinely stuck past ~5 min, restart session |
| Runtime disconnected mid-session ("Reconnecting...") | Colab's idle timeout (free tier) | Reconnect, then re-run from step 2 (or step 5B onward if using Drive) — everything before that point needs redoing |
