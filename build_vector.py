"""
build_vector.py
================
Builds the FAISS vector store for CycloneGPT.

Pipeline
--------
1.  Load the raw observation-level CSV files (one row per storm fix).
2.  Normalize columns and tag every row with its Region.
3.  Group all observations by cyclone NAME (+ Region) so that
    ONE cyclone == ONE LangChain Document. This is a deliberate design
    choice: users ask about cyclones as a whole ("Tell me about Cyclone
    Seroja"), not about a single 3-hourly fix, so document-per-cyclone
    retrieval produces far more coherent context than document-per-row.
4.  Deterministically compute a "Cyclone Profile" for each storm with
    pandas (no LLM involved - these are just descriptive statistics).
5.  Render the profile + the full chronological observation history into
    a single text block that becomes the Document's page_content.
6.  Embed every Document with a local HuggingFace sentence-transformer
    model and persist the resulting index to disk with FAISS so rag.py
    can load it later.

Run directly to (re)build the index:

    python build_vector.py

No API key is required for embedding (runs locally). A Gemini API key
(GOOGLE_API_KEY or GEMINI_API_KEY) is only needed later, by rag.py, for
the chat/generation step.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    # FAISS integration lives in langchain-community.
    from langchain_community.vectorstores import FAISS
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "langchain-community is required for the FAISS vector store. "
        "Install it with: pip install langchain-community"
    ) from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("build_vector")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

# Maps each source CSV to the human-readable Region label that gets
# stamped onto every row loaded from that file.
REGION_SOURCES: dict[str, str] = {
    "western_pacific.csv": "Western Pacific",
    "southeast_indian.csv": "Southeast Indian",
}

# The western_pacific.csv file ships with a typo in its header
# ("INTESINTY" instead of "INTENSITY"). We normalize every known column
# alias to a single canonical name so downstream code never has to think
# about which file a row came from.
COLUMN_ALIASES: dict[str, str] = {
    "INTESINTY": "INTENSITY",
    "INTENSITY": "INTENSITY",
    "NAME": "NAME",
    "TIME": "TIME",
    "LAT": "LAT",
    "LON": "LON",
    "STORM_SPEED": "STORM_SPEED",
    "STORM_DIR": "STORM_DIR",
}

REQUIRED_COLUMNS = ["NAME", "TIME", "LAT", "LON", "INTENSITY", "STORM_SPEED", "STORM_DIR"]

# Local HuggingFace sentence-transformer used for embeddings. Runs fully
# offline (no API key, no quota) - swapped in after GoogleGenerativeAI-
# Embeddings returned "403 Permission Denied" on this project's API key.
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# A cyclone must fluctuate by less than this fraction of its intensity
# range to be considered "flat" rather than trending. Purely a heuristic
# knob for classify_intensity_trend().
TREND_NOISE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_region_csv(filename: str, region: str) -> pd.DataFrame:
    """Load a single region CSV and normalize it into the canonical schema.

    Parameters
    ----------
    filename:
        File name (relative to DATA_DIR) of the raw CSV.
    region:
        Region label to attach to every row loaded from this file.

    Returns
    -------
    A DataFrame with columns REQUIRED_COLUMNS + ["REGION"], sorted
    chronologically within each cyclone.
    """
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Expected dataset '{filename}' not found in {DATA_DIR}. "
            "Make sure both CSV files are placed under CycloneGPT/data/."
        )

    # The source files are ';'-delimited, UTF-8 with a BOM, and use CRLF
    # line endings - encoding='utf-8-sig' strips the BOM automatically.
    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig")

    # Normalize column names (strip whitespace, fix known typos/aliases).
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"'{filename}' is missing expected column(s) {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["REGION"] = region

    # Types: TIME -> datetime, numeric columns -> numeric (coercing bad
    # values to NaN instead of raising, since some INTENSITY cells are
    # legitimately blank in the source data).
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
    for numeric_col in ("LAT", "LON", "INTENSITY", "STORM_SPEED", "STORM_DIR"):
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["NAME", "TIME", "LAT", "LON"])
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Dropped %d row(s) from '%s' with missing NAME/TIME/LAT/LON.",
            dropped,
            filename,
        )

    df["NAME"] = df["NAME"].astype(str).str.strip()
    df = df.sort_values(["NAME", "TIME"]).reset_index(drop=True)
    return df


def load_all_data() -> pd.DataFrame:
    """Load and concatenate every configured region CSV."""
    frames = [load_region_csv(fname, region) for fname, region in REGION_SOURCES.items()]
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d observations across %d region file(s).",
        len(combined),
        len(REGION_SOURCES),
    )
    return combined


# ---------------------------------------------------------------------------
# Cyclone profile computation (pure pandas, no LLM)
# ---------------------------------------------------------------------------

@dataclass
class CycloneProfile:
    """Deterministic, pandas-computed summary statistics for one cyclone.

    Every field here is derived directly from the data - nothing is
    inferred by an LLM. This keeps the factual backbone of each document
    trustworthy and cheap to regenerate.
    """

    name: str
    region: str
    unique_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_days: float
    total_observations: int
    max_intensity: float | None
    min_intensity: float | None
    avg_intensity: float | None
    max_storm_speed: float | None
    avg_storm_speed: float | None
    first_lat: float
    first_lon: float
    last_lat: float
    last_lon: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    initial_direction: float | None
    final_direction: float | None
    intensity_trend: str
    metadata: dict = field(default_factory=dict)


def classify_intensity_trend(intensity: pd.Series) -> str:
    """Classify a cyclone's intensity sequence as Increasing / Decreasing /
    Fluctuating / Data Unavailable.

    Approach
    --------
    1.  Drop missing readings. If fewer than 2 remain, the trend cannot be
        determined.
    2.  Fit a simple linear regression (slope) of intensity vs. observation
        index. The sign and relative magnitude of the slope tells us the
        *overall* direction of travel across the storm's lifetime.
    3.  Normalize the slope's total predicted change over the series by the
        intensity range. If the net change covers less than
        TREND_NOISE_THRESHOLD of the observed range, the storm is treated
        as "Fluctuating" (i.e., no dominant trend) rather than forcing a
        weak slope into Increasing/Decreasing.

    This is more robust than simply comparing the first and last readings,
    which is sensitive to noise at the very start/end of the observation
    window.
    """
    values = intensity.dropna().to_numpy(dtype=float)
    if values.size < 2:
        return "Data Unavailable"

    x = np.arange(values.size, dtype=float)
    slope, _ = np.polyfit(x, values, deg=1)

    value_range = values.max() - values.min()
    if value_range == 0:
        return "Fluctuating"

    net_predicted_change = slope * (values.size - 1)
    normalized_change = abs(net_predicted_change) / value_range

    if normalized_change < TREND_NOISE_THRESHOLD:
        return "Fluctuating"
    return "Increasing" if slope > 0 else "Decreasing"


def _safe_stat(series: pd.Series, func) -> float | None:
    """Apply an aggregation function while gracefully handling all-NaN
    columns (some cyclones have zero valid INTENSITY readings)."""
    clean = series.dropna()
    if clean.empty:
        return None
    return float(func(clean))


def build_cyclone_profile(name: str, region: str, group: pd.DataFrame) -> CycloneProfile:
    """Compute the full CycloneProfile for a single (NAME, REGION) group.

    `group` must already be sorted chronologically by TIME.
    """
    start_time = group["TIME"].iloc[0]
    end_time = group["TIME"].iloc[-1]
    duration_days = round((end_time - start_time).total_seconds() / 86400, 1)

    unique_id = f"{name}_{region.replace(' ', '_').upper()}"

    return CycloneProfile(
        name=name,
        region=region,
        unique_id=unique_id,
        start_time=start_time,
        end_time=end_time,
        duration_days=duration_days,
        total_observations=len(group),
        max_intensity=_safe_stat(group["INTENSITY"], np.max),
        min_intensity=_safe_stat(group["INTENSITY"], np.min),
        avg_intensity=_safe_stat(group["INTENSITY"], np.mean),
        max_storm_speed=_safe_stat(group["STORM_SPEED"], np.max),
        avg_storm_speed=_safe_stat(group["STORM_SPEED"], np.mean),
        first_lat=float(group["LAT"].iloc[0]),
        first_lon=float(group["LON"].iloc[0]),
        last_lat=float(group["LAT"].iloc[-1]),
        last_lon=float(group["LON"].iloc[-1]),
        lat_min=float(group["LAT"].min()),
        lat_max=float(group["LAT"].max()),
        lon_min=float(group["LON"].min()),
        lon_max=float(group["LON"].max()),
        initial_direction=_safe_stat(group["STORM_DIR"].iloc[:1], np.mean),
        final_direction=_safe_stat(group["STORM_DIR"].iloc[-1:], np.mean),
        intensity_trend=classify_intensity_trend(group["INTENSITY"]),
    )


# ---------------------------------------------------------------------------
# Text rendering (profile + observation history -> Document.page_content)
# ---------------------------------------------------------------------------

def _fmt(value: float | None, unit: str = "", precision: int = 1) -> str:
    """Format a numeric value for display, or say plainly that it's
    unavailable rather than printing 'None' or 'nan' into the document."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Data unavailable"
    if unit:
        return f"{value:.{precision}f} {unit}"
    return f"{value:.{precision}f}"


def _fmt_direction(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Data unavailable"
    return f"{value:.0f}\u00b0"


def render_profile_section(profile: CycloneProfile) -> str:
    """Render SECTION 1 - CYCLONE PROFILE as plain text."""
    lines = [
        "================================",
        "CYCLONE PROFILE",
        "================================",
        f"Start Date        : {profile.start_time:%Y-%m-%d %H:%M}",
        f"End Date          : {profile.end_time:%Y-%m-%d %H:%M}",
        f"Duration          : {profile.duration_days} Days",
        f"Total Observations: {profile.total_observations}",
        f"Maximum Intensity : {_fmt(profile.max_intensity, 'knots', 0)}",
        f"Minimum Intensity : {_fmt(profile.min_intensity, 'knots', 0)}",
        f"Average Intensity : {_fmt(profile.avg_intensity, 'knots')}",
        f"Maximum Storm Speed: {_fmt(profile.max_storm_speed)}",
        f"Average Storm Speed: {_fmt(profile.avg_storm_speed)}",
        f"First Position    : {profile.first_lat:.1f}, {profile.first_lon:.1f}",
        f"Last Position     : {profile.last_lat:.1f}, {profile.last_lon:.1f}",
        f"Latitude Range    : {profile.lat_min:.1f} - {profile.lat_max:.1f}",
        f"Longitude Range   : {profile.lon_min:.1f} - {profile.lon_max:.1f}",
        f"Initial Movement Direction: {_fmt_direction(profile.initial_direction)}",
        f"Final Movement Direction  : {_fmt_direction(profile.final_direction)}",
        f"Intensity Trend   : {profile.intensity_trend}",
    ]
    return "\n".join(lines)


def render_observation_history(group: pd.DataFrame) -> str:
    """Render SECTION 2 - OBSERVATION HISTORY as plain text, one line per
    chronological observation."""
    lines = [
        "================================",
        "OBSERVATION HISTORY",
        "================================",
    ]
    for _, row in group.iterrows():
        intensity = "N/A" if pd.isna(row["INTENSITY"]) else f"{row['INTENSITY']:.0f} knots"
        lines.append(
            f"{row['TIME']:%Y-%m-%d %H:%M} | "
            f"Lat {row['LAT']:.1f}, Lon {row['LON']:.1f} | "
            f"Intensity {intensity} | "
            f"Speed {row['STORM_SPEED']:.0f} | "
            f"Direction {row['STORM_DIR']:.0f}\u00b0"
        )
    return "\n".join(lines)


def build_document(name: str, region: str, group: pd.DataFrame) -> Document:
    """Build a single LangChain Document (profile + full history) for one
    cyclone, along with structured metadata for future filtering."""
    profile = build_cyclone_profile(name, region, group)

    header = f"Cyclone Name : {name}\nRegion : {region}"
    content = "\n\n".join(
        [
            header,
            render_profile_section(profile),
            render_observation_history(group),
        ]
    )

    metadata = {
        "name": profile.name,
        "region": profile.region,
        "unique_id": profile.unique_id,
        "start_date": profile.start_time.strftime("%Y-%m-%d"),
        "end_date": profile.end_time.strftime("%Y-%m-%d"),
        "start_year": int(profile.start_time.year),
        "end_year": int(profile.end_time.year),
        "duration_days": profile.duration_days,
        "observation_count": profile.total_observations,
        "max_intensity": profile.max_intensity,
        "min_intensity": profile.min_intensity,
        "avg_intensity": profile.avg_intensity,
        "intensity_trend": profile.intensity_trend,
    }

    return Document(page_content=content, metadata=metadata)


def build_documents(df: pd.DataFrame) -> list[Document]:
    """Group the full observation-level DataFrame by (NAME, REGION) and
    build one Document per cyclone.

    Grouping by (NAME, REGION) rather than NAME alone matters: the same
    storm name is reused across different basins/eras (e.g. "DOLLY" and
    "KEN 1" both appear in both regional datasets in this data), so NAME
    alone is not a unique cyclone identifier.
    """
    documents: list[Document] = []
    grouped = df.groupby(["NAME", "REGION"], sort=True)
    for (name, region), group in grouped:
        group = group.sort_values("TIME")
        documents.append(build_document(name, region, group))

    logger.info("Built %d cyclone document(s) from %d observation(s).", len(documents), len(df))
    return documents


# ---------------------------------------------------------------------------
# Embedding + FAISS persistence
# ---------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """Instantiate the local HuggingFace embedding model.

    Runs on-device via sentence-transformers - no API key or network call
    required, so there is no quota/permission failure mode here.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_and_persist_index(
    documents: Iterable[Document], index_dir: Path = FAISS_INDEX_DIR
) -> None:
    """Embed all documents and persist the resulting FAISS index to disk."""
    documents = list(documents)
    if not documents:
        raise ValueError("No documents were built - nothing to embed.")

    embeddings = get_embeddings()
    logger.info(
        "Embedding %d document(s) with HuggingFace (%s)...", len(documents), EMBEDDING_MODEL
    )
    vector_store = FAISS.from_documents(documents, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    logger.info("FAISS index persisted to '%s'.", index_dir)


def main() -> None:
    load_dotenv()  # pulls GOOGLE_API_KEY / GEMINI_API_KEY from a .env file if present
    try:
        raw_df = load_all_data()
        documents = build_documents(raw_df)
        build_and_persist_index(documents)
    except (FileNotFoundError, ValueError, EnvironmentError) as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
