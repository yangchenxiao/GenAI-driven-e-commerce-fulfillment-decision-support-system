import os
import re
import time
import json
from typing import Optional, Tuple, Dict, List

import pandas as pd
import streamlit as st
import mysql.connector
from dotenv import load_dotenv

import matplotlib.pyplot as plt


# =========================================================
# Gemini SDK (generation)
# =========================================================
try:
    from google import genai
except Exception:
    genai = None


# =========================================================
# RAG deps (LangChain + Chroma + local embeddings)
# =========================================================
RAG_IMPORT_OK = False
RAG_IMPORT_ERR = ""
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document

    # local embeddings (preferred; avoids Gemini embedding 404)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    RAG_IMPORT_OK = True
except Exception as e:
    RAG_IMPORT_OK = False
    RAG_IMPORT_ERR = f"{type(e).__name__}: {e}"


# =========================================================
# Setup
# =========================================================
load_dotenv()
st.set_page_config(page_title="Fulfillment Triage", layout="wide")
st.title("Fulfillment Triage (SQL check)")


# =========================================================
# Env config
# =========================================================
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()

# RAG config
CHROMA_DIR = os.getenv("CHROMA_DIR", "kb_index").strip()
POLICY_PATH = os.getenv("POLICY_PATH", "kb/policy.md").strip()
OPS_KB_PATH = os.getenv("OPS_KB_PATH", "kb/ops_kb.md").strip()
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()

# MySQL
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "olist")


# =========================================================
# DB helpers
# =========================================================
def get_db_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        autocommit=True,
    )


def run_query(sql: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def table_exists(table_name: str) -> bool:
    q = f"""
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = '{MYSQL_DB}'
      AND table_name = '{table_name}'
    LIMIT 1;
    """.strip()
    try:
        df = run_query(q)
        return not df.empty
    except Exception:
        return False


# =========================================================
# Gemini helpers + 429 cooldown
# =========================================================
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


client = get_gemini_client()


def _extract_retry_seconds(err_text: str) -> int:
    """
    Try to parse retryDelay seconds from Gemini error payload.
    Examples seen:
      - "retryDelay": "51s"
      - "Please retry in 51.56s"
    """
    if not err_text:
        return 60
    m = re.search(r"retryDelay[^0-9]*([0-9]+)\s*s", err_text, flags=re.IGNORECASE)
    if m:
        return max(5, int(m.group(1)))
    m = re.search(r"retry in\s*([0-9]+(?:\.[0-9]+)?)\s*s", err_text, flags=re.IGNORECASE)
    if m:
        return max(5, int(float(m.group(1))))
    return 60


def _is_quota_error(err_text: str) -> bool:
    t = (err_text or "").lower()
    return ("429" in t) or ("resource_exhausted" in t) or ("quota" in t) or ("rate limit" in t)


def gemini_text(prompt: str) -> str:
    if client is None:
        raise RuntimeError("Gemini client unavailable")
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return (resp.text or "").strip()


def gemini_json(prompt: str) -> dict:
    raw = gemini_text(prompt).strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


# =========================================================
# Policy loader + RAG retriever
# =========================================================
def load_policy_text() -> Optional[str]:
    if not POLICY_PATH or (not os.path.exists(POLICY_PATH)):
        return None
    try:
        with open(POLICY_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

def load_ops_kb_text() -> Optional[str]:
    if not OPS_KB_PATH or (not os.path.exists(OPS_KB_PATH)):
        return None
    try:
        with open(OPS_KB_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def build_policy_retriever() -> Tuple[Optional[object], str]:
    """
    Returns (retriever, status_str)
    Build ONE merged retriever for: policy.md + ops_kb.md
    """
    policy_text = load_policy_text() or ""
    ops_text = load_ops_kb_text() or ""

    if not policy_text and not ops_text:
        return None, "disabled"

    if not RAG_IMPORT_OK:
        return None, f"disabled ({RAG_IMPORT_ERR})"

    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n## ", "\n# ", "\n\n", "\n", " "],
        )

        docs = []
        if policy_text.strip():
            chunks = [c.strip() for c in splitter.split_text(policy_text) if c.strip()]
            docs.extend([Document(page_content=c, metadata={"source": "policy", "path": POLICY_PATH}) for c in chunks])

        if ops_text.strip():
            chunks = [c.strip() for c in splitter.split_text(ops_text) if c.strip()]
            docs.extend([Document(page_content=c, metadata={"source": "ops_kb", "path": OPS_KB_PATH}) for c in chunks])

        embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

        vectordb = Chroma(
            collection_name="ops_knowledge_rag",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

        existing = 0
        try:
            existing = vectordb._collection.count()
        except Exception:
            existing = 0

        if existing == 0:
            vectordb.add_documents(docs)

        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        status_bits = []
        if policy_text.strip():
            status_bits.append("policy")
        if ops_text.strip():
            status_bits.append("ops_kb")
        return retriever, "enabled (" + "+".join(status_bits) + ")"

    except Exception as e:
        return None, f"disabled ({type(e).__name__}: {e})"

policy_retriever, policy_rag_status = build_policy_retriever()


def retrieve_policy_context(query: str) -> str:
    """
    Unified retrieval over merged KB (policy + ops_kb).
    """
    policy_text = load_policy_text() or ""
    ops_text = load_ops_kb_text() or ""
    fallback_text = (policy_text + "\n\n" + ops_text).strip()

    if policy_retriever is None:
        return fallback_text[:1500].strip()

    try:
        try:
            docs = policy_retriever.invoke(query)
        except Exception:
            docs = policy_retriever.get_relevant_documents(query)

        out = []
        for i, d in enumerate(docs, 1):
            src = (d.metadata or {}).get("source", "kb")
            out.append(f"[KB chunk {i} | {src}]\n{d.page_content.strip()}")
        return "\n\n".join(out).strip()
    except Exception:
        return fallback_text[:1500].strip()


# =========================================================
# Parameters
# =========================================================
st.subheader("Parameters")
weeks = st.selectbox("Time window (weeks)", [8, 12, 26, 52], index=2)
min_orders = st.number_input("Minimum orders per category", 10, 5000, 200, 10)
top_k = st.number_input("Top K categories", 5, 50, 20, 5)


# =========================================================
# SQL tools
# =========================================================
def sql_late_delivery(weeks_: int, min_orders_: int, top_k_: int) -> str:
    return f"""
WITH lastN AS (
  SELECT *
  FROM orders
  WHERE order_purchase_timestamp >= DATE_SUB(
    (SELECT MAX(order_purchase_timestamp) FROM orders),
    INTERVAL {weeks_} WEEK
  )
),
delivered AS (
  SELECT *
  FROM lastN
  WHERE order_status = 'delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_estimated_delivery_date IS NOT NULL
)
SELECT
  vi.category_en,
  COUNT(DISTINCT d.order_id) AS n_orders,
  SUM(CASE
        WHEN d.order_delivered_customer_date > d.order_estimated_delivery_date
        THEN 1 ELSE 0 END) AS n_late,
  AVG(CASE
        WHEN d.order_delivered_customer_date > d.order_estimated_delivery_date
        THEN 1 ELSE 0 END) AS late_rate,
  AVG(CASE
        WHEN d.order_delivered_customer_date > d.order_estimated_delivery_date
        THEN TIMESTAMPDIFF(DAY, d.order_estimated_delivery_date, d.order_delivered_customer_date)
        ELSE NULL END) AS avg_days_late
FROM delivered d
JOIN v_items_en vi ON vi.order_id = d.order_id
GROUP BY vi.category_en
HAVING n_orders >= {min_orders_}
ORDER BY n_late DESC, late_rate DESC, n_orders DESC
LIMIT {top_k_};
""".strip()


def sql_stage_clue(weeks_: int, min_orders_: int, top_k_: int) -> str:
    return f"""
WITH lastN AS (
  SELECT *
  FROM orders
  WHERE order_purchase_timestamp >= DATE_SUB(
    (SELECT MAX(order_purchase_timestamp) FROM orders),
    INTERVAL {weeks_} WEEK
  )
),
delivered AS (
  SELECT *
  FROM lastN
  WHERE order_status = 'delivered'
    AND order_delivered_customer_date IS NOT NULL
)
SELECT
  vi.category_en,
  COUNT(DISTINCT d.order_id) AS n_delivered,
  AVG(TIMESTAMPDIFF(HOUR, d.order_purchase_timestamp, d.order_approved_at)) AS avg_h_purchase_to_approved,
  AVG(TIMESTAMPDIFF(HOUR, d.order_approved_at, d.order_delivered_carrier_date)) AS avg_h_approved_to_carrier,
  AVG(TIMESTAMPDIFF(HOUR, d.order_delivered_carrier_date, d.order_delivered_customer_date)) AS avg_h_carrier_to_customer
FROM delivered d
JOIN v_items_en vi ON vi.order_id = d.order_id
WHERE d.order_approved_at IS NOT NULL
  AND d.order_delivered_carrier_date IS NOT NULL
GROUP BY vi.category_en
HAVING n_delivered >= {min_orders_}
ORDER BY avg_h_approved_to_carrier DESC
LIMIT {top_k_};
""".strip()


def sql_review_risk(weeks_: int, min_orders_: int, top_k_: int) -> str:
    return f"""
WITH lastN AS (
  SELECT *
  FROM orders
  WHERE order_purchase_timestamp >= DATE_SUB(
    (SELECT MAX(order_purchase_timestamp) FROM orders),
    INTERVAL {weeks_} WEEK
  )
),
joined AS (
  SELECT
    o.order_id,
    o.order_status,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    r.review_score
  FROM lastN o
  LEFT JOIN order_reviews r ON r.order_id = o.order_id
),
delivered AS (
  SELECT *
  FROM joined
  WHERE order_status='delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_estimated_delivery_date IS NOT NULL
)
SELECT
  vi.category_en,
  COUNT(DISTINCT d.order_id) AS n_orders,
  SUM(CASE WHEN d.review_score IS NOT NULL AND d.review_score <= 2 THEN 1 ELSE 0 END) AS n_low_ratings,
  AVG(CASE WHEN d.review_score IS NOT NULL AND d.review_score <= 2 THEN 1 ELSE 0 END) AS low_rating_rate,
  AVG(CASE WHEN d.review_score IS NOT NULL THEN d.review_score ELSE NULL END) AS avg_review_score,
  SUM(CASE WHEN d.order_delivered_customer_date > d.order_estimated_delivery_date THEN 1 ELSE 0 END) AS n_late,
  AVG(CASE WHEN d.order_delivered_customer_date > d.order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_rate
FROM delivered d
JOIN v_items_en vi ON vi.order_id = d.order_id
GROUP BY vi.category_en
HAVING n_orders >= {min_orders_}
ORDER BY n_low_ratings DESC, low_rating_rate DESC, n_orders DESC
LIMIT {top_k_};
""".strip()


SQL_TOOLS = {
    "late_delivery": ("Late delivery by category", sql_late_delivery),
    "stage_clue": ("Pre-carrier vs in-transit stage clues", sql_stage_clue),
    "review_risk": ("Review risk by category", sql_review_risk),
}


# =========================================================
# Deterministic drill-down builder (packed SQL: main + fallback)
# =========================================================
HAS_ORDER_ITEMS = table_exists("order_items")
HAS_CUSTOMERS = table_exists("customers")
HAS_SELLERS = table_exists("sellers")


def build_next_query_sql(intent: str, df: pd.DataFrame) -> Tuple[str, str]:
    cat_list: List[str] = []
    if df is not None and (not df.empty) and ("category_en" in df.columns):
        cat_list = df["category_en"].head(1).astype(str).tolist()

    drill_min_orders = max(20, int(min_orders) // 10)

    if not (HAS_ORDER_ITEMS and HAS_CUSTOMERS and HAS_SELLERS and cat_list):
        return "Robustness check: change weeks/min_orders and rerun", "-- Change weeks/min_orders/top_k in UI and rerun."

    cat = cat_list[0].replace("'", "''")

    if intent in ("late_delivery", "review_risk"):
        title = f"Drill down lanes (seller_state → customer_state) for category='{cat}'"

        sql_with_having = f"""
WITH lastN AS (
  SELECT *
  FROM orders
  WHERE order_purchase_timestamp >= DATE_SUB(
    (SELECT MAX(order_purchase_timestamp) FROM orders),
    INTERVAL {weeks} WEEK
  )
),
delivered AS (
  SELECT *
  FROM lastN
  WHERE order_status='delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_estimated_delivery_date IS NOT NULL
),
base AS (
  SELECT
    vi.order_id,
    vi.category_en,
    d.customer_id,
    oi.seller_id,
    d.order_delivered_customer_date,
    d.order_estimated_delivery_date
  FROM v_items_en vi
  JOIN delivered d ON d.order_id = vi.order_id
  JOIN order_items oi ON oi.order_id = vi.order_id
  WHERE vi.category_en = '{cat}'
),
with_dim AS (
  SELECT
    b.*,
    COALESCE(c.customer_state, 'NA') AS customer_state,
    COALESCE(s.seller_state, 'NA') AS seller_state
  FROM base b
  LEFT JOIN customers c ON c.customer_id = b.customer_id
  LEFT JOIN sellers s ON s.seller_id = b.seller_id
)
SELECT
  seller_state,
  customer_state,
  COUNT(DISTINCT order_id) AS n_orders,
  SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS n_late,
  AVG(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_rate,
  AVG(CASE
        WHEN order_delivered_customer_date > order_estimated_delivery_date
        THEN TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date)
        ELSE NULL END) AS avg_days_late
FROM with_dim
GROUP BY seller_state, customer_state
HAVING n_orders >= {drill_min_orders}
ORDER BY n_late DESC, late_rate DESC, n_orders DESC
LIMIT 30;
""".strip()

        sql_no_having = f"""
WITH lastN AS (
  SELECT *
  FROM orders
  WHERE order_purchase_timestamp >= DATE_SUB(
    (SELECT MAX(order_purchase_timestamp) FROM orders),
    INTERVAL {weeks} WEEK
  )
),
delivered AS (
  SELECT *
  FROM lastN
  WHERE order_status='delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_estimated_delivery_date IS NOT NULL
),
base AS (
  SELECT
    vi.order_id,
    vi.category_en,
    d.customer_id,
    oi.seller_id,
    d.order_delivered_customer_date,
    d.order_estimated_delivery_date
  FROM v_items_en vi
  JOIN delivered d ON d.order_id = vi.order_id
  JOIN order_items oi ON oi.order_id = vi.order_id
  WHERE vi.category_en = '{cat}'
),
with_dim AS (
  SELECT
    b.*,
    COALESCE(c.customer_state, 'NA') AS customer_state,
    COALESCE(s.seller_state, 'NA') AS seller_state
  FROM base b
  LEFT JOIN customers c ON c.customer_id = b.customer_id
  LEFT JOIN sellers s ON s.seller_id = b.seller_id
)
SELECT
  seller_state,
  customer_state,
  COUNT(DISTINCT order_id) AS n_orders,
  SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS n_late,
  AVG(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_rate,
  AVG(CASE
        WHEN order_delivered_customer_date > order_estimated_delivery_date
        THEN TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date)
        ELSE NULL END) AS avg_days_late
FROM with_dim
GROUP BY seller_state, customer_state
ORDER BY n_late DESC, n_orders DESC
LIMIT 30;
""".strip()

        packed = sql_with_having + "\n\n-- __FALLBACK_NO_HAVING__\n\n" + sql_no_having
        return title, packed

    if intent == "stage_clue":
        title = f"In-transit hours by lane for category='{cat}'"

        sql_with_having = f"""
WITH lastN AS (
  SELECT *
  FROM orders
  WHERE order_purchase_timestamp >= DATE_SUB(
    (SELECT MAX(order_purchase_timestamp) FROM orders),
    INTERVAL {weeks} WEEK
  )
),
delivered AS (
  SELECT *
  FROM lastN
  WHERE order_status='delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_delivered_carrier_date IS NOT NULL
    AND order_approved_at IS NOT NULL
),
base AS (
  SELECT
    vi.order_id,
    vi.category_en,
    d.customer_id,
    oi.seller_id,
    TIMESTAMPDIFF(HOUR, d.order_purchase_timestamp, d.order_approved_at) AS h_purchase_to_approved,
    TIMESTAMPDIFF(HOUR, d.order_approved_at, d.order_delivered_carrier_date) AS h_approved_to_carrier,
    TIMESTAMPDIFF(HOUR, d.order_delivered_carrier_date, d.order_delivered_customer_date) AS h_carrier_to_customer
  FROM v_items_en vi
  JOIN delivered d ON d.order_id = vi.order_id
  JOIN order_items oi ON oi.order_id = vi.order_id
  WHERE vi.category_en = '{cat}'
),
with_dim AS (
  SELECT
    b.*,
    COALESCE(c.customer_state, 'NA') AS customer_state,
    COALESCE(s.seller_state, 'NA') AS seller_state
  FROM base b
  LEFT JOIN customers c ON c.customer_id = b.customer_id
  LEFT JOIN sellers s ON s.seller_id = b.seller_id
)
SELECT
  seller_state,
  customer_state,
  COUNT(DISTINCT order_id) AS n_delivered,
  AVG(h_purchase_to_approved) AS avg_h_purchase_to_approved,
  AVG(h_approved_to_carrier) AS avg_h_approved_to_carrier,
  AVG(h_carrier_to_customer) AS avg_h_carrier_to_customer
FROM with_dim
GROUP BY seller_state, customer_state
HAVING n_delivered >= {drill_min_orders}
ORDER BY avg_h_carrier_to_customer DESC, n_delivered DESC
LIMIT 30;
""".strip()

        sql_no_having = f"""
WITH lastN AS (
  SELECT *
  FROM orders
  WHERE order_purchase_timestamp >= DATE_SUB(
    (SELECT MAX(order_purchase_timestamp) FROM orders),
    INTERVAL {weeks} WEEK
  )
),
delivered AS (
  SELECT *
  FROM lastN
  WHERE order_status='delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_delivered_carrier_date IS NOT NULL
    AND order_approved_at IS NOT NULL
),
base AS (
  SELECT
    vi.order_id,
    vi.category_en,
    d.customer_id,
    oi.seller_id,
    TIMESTAMPDIFF(HOUR, d.order_purchase_timestamp, d.order_approved_at) AS h_purchase_to_approved,
    TIMESTAMPDIFF(HOUR, d.order_approved_at, d.order_delivered_carrier_date) AS h_approved_to_carrier,
    TIMESTAMPDIFF(HOUR, d.order_delivered_carrier_date, d.order_delivered_customer_date) AS h_carrier_to_customer
  FROM v_items_en vi
  JOIN delivered d ON d.order_id = vi.order_id
  JOIN order_items oi ON oi.order_id = vi.order_id
  WHERE vi.category_en = '{cat}'
),
with_dim AS (
  SELECT
    b.*,
    COALESCE(c.customer_state, 'NA') AS customer_state,
    COALESCE(s.seller_state, 'NA') AS seller_state
  FROM base b
  LEFT JOIN customers c ON c.customer_id = b.customer_id
  LEFT JOIN sellers s ON s.seller_id = b.seller_id
)
SELECT
  seller_state,
  customer_state,
  COUNT(DISTINCT order_id) AS n_delivered,
  AVG(h_purchase_to_approved) AS avg_h_purchase_to_approved,
  AVG(h_approved_to_carrier) AS avg_h_approved_to_carrier,
  AVG(h_carrier_to_customer) AS avg_h_carrier_to_customer
FROM with_dim
GROUP BY seller_state, customer_state
ORDER BY n_delivered DESC
LIMIT 30;
""".strip()

        packed = sql_with_having + "\n\n-- __FALLBACK_NO_HAVING__\n\n" + sql_no_having
        return title, packed

    return "Robustness check: change weeks/min_orders and rerun", "-- Change weeks/min_orders/top_k in UI and rerun."


# =========================================================
# Stage 4.3 — Executable multi-step Diagnosis Plan
# =========================================================
def build_diagnosis_plan(intent: str, preview_df: pd.DataFrame) -> List[Dict[str, object]]:
    top_cat = None
    if preview_df is not None and (not preview_df.empty) and ("category_en" in preview_df.columns):
        top_cat = str(preview_df.iloc[0]["category_en"])
    cat = top_cat or "top category"

    if intent == "late_delivery":
        return [
            {
                "title": "Step 1 - Confirm impact",
                "purpose": "Confirm this is high-impact (volume × late volume), not only a high rate.",
                "action": f"Validate `{cat}` using `n_orders`, `n_late`, `late_rate`, `avg_days_late` in the preview table.",
                "tool": "mark_checked",
                "params": {"category": cat, "intent": intent},
            },
            {
                "title": "Step 2 — Localize by lane",
                "purpose": "Check whether delays concentrate in a few seller→customer lanes (actionable routing).",
                "action": "Run lane drill-down (seller_state → customer_state).",
                "tool": "lane_drilldown",
                "params": {"category": cat, "intent": intent},
            },
            {
                "title": "Step 3 — Trend sanity check",
                "purpose": "Confirm the signal is stable over time (avoid one-off spikes).",
                "action": "Run category weekly trend on late volume + late rate.",
                "tool": "weekly_trend",
                "params": {"category": cat, "intent": intent},
            },
        ]

    if intent == "stage_clue":
        return [
            {
                "title": "Step 1 — Identify dominant stage",
                "purpose": "Decide whether the symptom skews pre-carrier or in-transit (clue, not root cause).",
                "action": f"Compare `{cat}` using `avg_h_approved_to_carrier` vs `avg_h_carrier_to_customer` in the preview.",
                "tool": "mark_checked",
                "params": {"category": cat, "intent": intent},
            },
            {
                "title": "Step 2 — Localize by lane",
                "purpose": "Check whether the stage gap is concentrated in specific lanes.",
                "action": "Run lane drill-down (lane-level stage timings).",
                "tool": "lane_drilldown",
                "params": {"category": cat, "intent": intent},
            },
            {
                "title": "Step 3 — Trend sanity check",
                "purpose": "Check whether the stage gap persists under time slicing.",
                "action": "Run category weekly trend on stage timings.",
                "tool": "weekly_trend",
                "params": {"category": cat, "intent": intent},
            },
        ]

    # review_risk
    return [
        {
            "title": "Step 1 — Quantify risk exposure",
            "purpose": "Prioritize by impact (low-rating volume), not only by rate.",
            "action": f"Validate `{cat}` using `n_orders`, `n_low_ratings`, `low_rating_rate`, `avg_review_score` in the preview.",
            "tool": "mark_checked",
            "params": {"category": cat, "intent": intent},
        },
        {
            "title": "Step 2 — Localize by lane",
            "purpose": "Check whether low ratings cluster in specific lanes (possible service/fulfillment issue).",
            "action": "Run lane drill-down (seller_state → customer_state).",
            "tool": "lane_drilldown",
            "params": {"category": cat, "intent": intent},
        },
        {
            "title": "Step 3 — Cross-signal trend",
            "purpose": "Check if low ratings co-move with delivery performance over time (correlation ≠ causation).",
            "action": "Run category weekly trend on low ratings + late rate.",
            "tool": "weekly_trend",
            "params": {"category": cat, "intent": intent},
        },
    ]


def _sql_lane_drilldown(intent: str, category: str) -> str:
    cat = (category or "").replace("'", "''")
    drill_min_orders = max(20, int(min_orders) // 10)

    if intent in ("late_delivery", "review_risk"):
        return f"""
WITH lastN AS (
  SELECT *
  FROM orders
  WHERE order_purchase_timestamp >= DATE_SUB(
    (SELECT MAX(order_purchase_timestamp) FROM orders),
    INTERVAL {weeks} WEEK
  )
),
delivered AS (
  SELECT *
  FROM lastN
  WHERE order_status='delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_estimated_delivery_date IS NOT NULL
),
base AS (
  SELECT
    vi.order_id,
    d.customer_id,
    oi.seller_id,
    d.order_delivered_customer_date,
    d.order_estimated_delivery_date
  FROM v_items_en vi
  JOIN delivered d ON d.order_id = vi.order_id
  JOIN order_items oi ON oi.order_id = vi.order_id
  WHERE vi.category_en = '{cat}'
),
with_dim AS (
  SELECT
    b.*,
    COALESCE(c.customer_state, 'NA') AS customer_state,
    COALESCE(s.seller_state, 'NA') AS seller_state
  FROM base b
  LEFT JOIN customers c ON c.customer_id = b.customer_id
  LEFT JOIN sellers s ON s.seller_id = b.seller_id
)
SELECT
  seller_state,
  customer_state,
  COUNT(DISTINCT order_id) AS n_orders,
  SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS n_late,
  AVG(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_rate,
  AVG(CASE
        WHEN order_delivered_customer_date > order_estimated_delivery_date
        THEN TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date)
        ELSE NULL END) AS avg_days_late
FROM with_dim
GROUP BY seller_state, customer_state
HAVING n_orders >= {drill_min_orders}
ORDER BY n_late DESC, late_rate DESC, n_orders DESC
LIMIT 30;
""".strip()

    if intent == "stage_clue":
        return f"""
WITH lastN AS (
  SELECT *
  FROM orders
  WHERE order_purchase_timestamp >= DATE_SUB(
    (SELECT MAX(order_purchase_timestamp) FROM orders),
    INTERVAL {weeks} WEEK
  )
),
delivered AS (
  SELECT *
  FROM lastN
  WHERE order_status='delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_delivered_carrier_date IS NOT NULL
    AND order_approved_at IS NOT NULL
),
base AS (
  SELECT
    vi.order_id,
    d.customer_id,
    oi.seller_id,
    TIMESTAMPDIFF(HOUR, d.order_purchase_timestamp, d.order_approved_at) AS h_purchase_to_approved,
    TIMESTAMPDIFF(HOUR, d.order_approved_at, d.order_delivered_carrier_date) AS h_approved_to_carrier,
    TIMESTAMPDIFF(HOUR, d.order_delivered_carrier_date, d.order_delivered_customer_date) AS h_carrier_to_customer
  FROM v_items_en vi
  JOIN delivered d ON d.order_id = vi.order_id
  JOIN order_items oi ON oi.order_id = vi.order_id
  WHERE vi.category_en = '{cat}'
),
with_dim AS (
  SELECT
    b.*,
    COALESCE(c.customer_state, 'NA') AS customer_state,
    COALESCE(s.seller_state, 'NA') AS seller_state
  FROM base b
  LEFT JOIN customers c ON c.customer_id = b.customer_id
  LEFT JOIN sellers s ON s.seller_id = b.seller_id
)
SELECT
  seller_state,
  customer_state,
  COUNT(DISTINCT order_id) AS n_delivered,
  AVG(h_purchase_to_approved) AS avg_h_purchase_to_approved,
  AVG(h_approved_to_carrier) AS avg_h_approved_to_carrier,
  AVG(h_carrier_to_customer) AS avg_h_carrier_to_customer
FROM with_dim
GROUP BY seller_state, customer_state
HAVING n_delivered >= {drill_min_orders}
ORDER BY avg_h_carrier_to_customer DESC, n_delivered DESC
LIMIT 30;
""".strip()

    return "-- unsupported intent for lane drilldown"


def _sql_weekly_trend(intent: str, category: str) -> str:
    cat = (category or "").replace("'", "''")

    if intent in ("late_delivery", "review_risk"):
        return f"""
WITH scoped AS (
  SELECT
    o.order_id,
    o.order_purchase_timestamp,
    o.order_status,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date
  FROM orders o
  JOIN v_items_en vi ON vi.order_id = o.order_id
  WHERE vi.category_en = '{cat}'
    AND o.order_purchase_timestamp >= DATE_SUB(
      (SELECT MAX(order_purchase_timestamp) FROM orders),
      INTERVAL {weeks} WEEK
    )
),
delivered AS (
  SELECT *
  FROM scoped
  WHERE order_status='delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_estimated_delivery_date IS NOT NULL
)
SELECT
  DATE_FORMAT(DATE_SUB(order_purchase_timestamp, INTERVAL WEEKDAY(order_purchase_timestamp) DAY), '%Y-%m-%d') AS week_start,
  COUNT(DISTINCT order_id) AS n_orders,
  SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS n_late,
  AVG(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_rate,
  AVG(CASE
        WHEN order_delivered_customer_date > order_estimated_delivery_date
        THEN TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date)
        ELSE NULL END) AS avg_days_late
FROM delivered
GROUP BY week_start
ORDER BY week_start ASC;
""".strip()

    if intent == "stage_clue":
        return f"""
WITH scoped AS (
  SELECT
    o.order_id,
    o.order_purchase_timestamp,
    o.order_status,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date
  FROM orders o
  JOIN v_items_en vi ON vi.order_id = o.order_id
  WHERE vi.category_en = '{cat}'
    AND o.order_purchase_timestamp >= DATE_SUB(
      (SELECT MAX(order_purchase_timestamp) FROM orders),
      INTERVAL {weeks} WEEK
    )
),
delivered AS (
  SELECT *
  FROM scoped
  WHERE order_status='delivered'
    AND order_delivered_customer_date IS NOT NULL
    AND order_delivered_carrier_date IS NOT NULL
    AND order_approved_at IS NOT NULL
)
SELECT
  DATE_FORMAT(DATE_SUB(order_purchase_timestamp, INTERVAL WEEKDAY(order_purchase_timestamp) DAY), '%Y-%m-%d') AS week_start,
  COUNT(DISTINCT order_id) AS n_delivered,
  AVG(TIMESTAMPDIFF(HOUR, order_purchase_timestamp, order_approved_at)) AS avg_h_purchase_to_approved,
  AVG(TIMESTAMPDIFF(HOUR, order_approved_at, order_delivered_carrier_date)) AS avg_h_approved_to_carrier,
  AVG(TIMESTAMPDIFF(HOUR, order_delivered_carrier_date, order_delivered_customer_date)) AS avg_h_carrier_to_customer
FROM delivered
GROUP BY week_start
ORDER BY week_start ASC;
""".strip()

    return "-- unsupported intent for weekly trend"


def run_plan_step(step: Dict[str, object], *, plan_sig: str, step_idx: int) -> Tuple[Optional[pd.DataFrame], str]:
    tool = str(step.get("tool", "noop"))
    params = step.get("params") if isinstance(step.get("params"), dict) else {}

    intent = str(params.get("intent", ""))
    category = str(params.get("category", ""))

    if tool == "mark_checked":
        st.session_state.setdefault("plan_checked", {})
        checked_list = st.session_state["plan_checked"].get(plan_sig, [])
        if not isinstance(checked_list, list):
            checked_list = list(checked_list)

        if step_idx not in checked_list:
            checked_list.append(step_idx)

        st.session_state["plan_checked"][plan_sig] = checked_list
        return None, "Marked as checked."

    if tool == "noop":
        return None, "This step is a checklist (no query to run)."

    if tool in ("lane_drilldown",) and not (HAS_ORDER_ITEMS and HAS_CUSTOMERS and HAS_SELLERS):
        return None, "Lane drill-down requires tables: order_items, customers, sellers."

    if tool == "lane_drilldown":
        sql = _sql_lane_drilldown(intent=intent, category=category)
        df = run_query(sql)
        return df, "Lane drill-down executed."

    if tool == "weekly_trend":
        sql = _sql_weekly_trend(intent=intent, category=category)
        df = run_query(sql)
        return df, "Weekly trend executed."

    return None, f"Unknown plan tool: {tool}"


# =========================================================
# Routing
# =========================================================
def route_intent_rulebased(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["late", "delay", "delayed", "on-time", "on time", "lateness", "delivery"]):
        return "late_delivery"
    if any(k in t for k in ["pre-carrier", "in-transit", "in transit", "stage", "carrier", "shipment", "handover"]):
        return "stage_clue"
    if any(k in t for k in ["review", "rating", "score", "low rating", "bad review", "satisfaction"]):
        return "review_risk"
    return "unknown"


def route_intent(text: str) -> Tuple[str, str]:
    if client is None:
        return route_intent_rulebased(text), "fallback"

    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["late_delivery", "stage_clue", "review_risk", "unknown"]},
            "confidence": {"type": "number"},
        },
        "required": ["intent", "confidence"],
        "additionalProperties": False,
    }

    prompt = f"""
You are routing user questions for an e-commerce fulfillment triage chatbot.
Pick exactly one intent:
- late_delivery: late delivery rate, on-time vs late, late categories
- stage_clue: pre-carrier vs in-transit, stage delays, carrier handoff timing
- review_risk: review score, low rating share, satisfaction risk

Return ONLY valid JSON matching this schema:
{json.dumps(schema)}

User question:
{text}
""".strip()

    try:
        data = gemini_json(prompt)
        intent = data.get("intent", "unknown")
        conf = float(data.get("confidence", 0))
        if intent not in SQL_TOOLS and intent != "unknown":
            return "unknown", "llm"
        if conf < 0.55:
            return route_intent_rulebased(text), "fallback"
        return intent, "llm"
    except Exception:
        return route_intent_rulebased(text), "fallback"


# =========================================================
# Chat render + Plan render (panel)
# =========================================================
def _render_lane_chart(df: pd.DataFrame):
    """
    Ops chart: Top lanes bar(n_late) + line(late_rate) on secondary axis.
    Requires seller_state, customer_state, n_late, late_rate.
    """
    need = {"seller_state", "customer_state", "n_late", "late_rate"}
    if not need.issubset(set(df.columns)):
        return

    dff = df.copy()
    dff["lane"] = dff["seller_state"].astype(str) + "→" + dff["customer_state"].astype(str)
    dff["n_late"] = pd.to_numeric(dff["n_late"], errors="coerce")
    dff["late_rate"] = pd.to_numeric(dff["late_rate"], errors="coerce")
    dff = dff.dropna(subset=["n_late"]).sort_values("n_late", ascending=False).head(10)
    if dff.empty:
        return

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(dff["lane"], dff["n_late"])
    ax2.plot(dff["lane"], dff["late_rate"], marker="o")

    ax1.set_xlabel("lane (seller→customer)")
    ax1.set_ylabel("n_late")
    ax2.set_ylabel("late_rate")

    ax1.set_title("Top lanes (n_late) with late_rate overlay")
    ax1.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)


def _render_weekly_dual_axis(df: pd.DataFrame):
    """
    Ops chart:
    - If has week_start + n_late + late_rate: bar(n_late) + line(late_rate) dual-axis
    - Else if stage timing trend: line two stage metrics
    """
    if "week_start" not in df.columns:
        return

    dff = df.copy()
    dff["week_start"] = pd.to_datetime(dff["week_start"], errors="coerce")
    dff = dff.dropna(subset=["week_start"]).sort_values("week_start")
    if dff.empty:
        return

    # common numeric conversion
    for c in [
        "n_late", "late_rate", "n_orders", "avg_days_late",
        "n_delivered", "avg_h_approved_to_carrier", "avg_h_carrier_to_customer"
    ]:
        if c in dff.columns:
            dff[c] = pd.to_numeric(dff[c], errors="coerce")

    if ("n_late" in dff.columns) and ("late_rate" in dff.columns):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.bar(dff["week_start"], dff["n_late"])
        ax2.plot(dff["week_start"], dff["late_rate"], marker="o")

        ax1.set_xlabel("week_start")
        ax1.set_ylabel("n_late")
        ax2.set_ylabel("late_rate")

        ax1.set_title("Weekly trend: n_late (bar) + late_rate (line)")
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        return

    if ("avg_h_approved_to_carrier" in dff.columns) and ("avg_h_carrier_to_customer" in dff.columns):
        fig, ax = plt.subplots()
        ax.plot(dff["week_start"], dff["avg_h_approved_to_carrier"], marker="o", label="avg_h_approved_to_carrier")
        ax.plot(dff["week_start"], dff["avg_h_carrier_to_customer"], marker="o", label="avg_h_carrier_to_customer")
        ax.set_xlabel("week_start")
        ax.set_ylabel("hours")
        ax.set_title("Weekly trend: stage timing (hours)")
        ax.legend()
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        return


def render_diagnosis_plan(plan: List[Dict[str, object]]):
    if not plan:
        st.info("No diagnosis plan yet. Ask a question above to generate one.")
        return

    st.markdown("### Diagnosis plan")
    st.caption("Executable 3-step workflow (domain-style).")

    # plan_sig fallback: if missing, compute from current plan
    plan_sig = st.session_state.get("latest_plan_sig", "")
    if not plan_sig:
        plan_sig = json.dumps(plan, ensure_ascii=False, sort_keys=True)
        st.session_state["latest_plan_sig"] = plan_sig

    checked_list = st.session_state.get("plan_checked", {}).get(plan_sig, [])
    if not isinstance(checked_list, list):
        checked_list = list(checked_list)
    checked_set = set(checked_list)

    for idx, step in enumerate(plan, 1):
        title = str(step.get("title", "")).strip()
        purpose = str(step.get("purpose", "")).strip()
        action = str(step.get("action", "")).strip()
        tool = str(step.get("tool", "noop")).strip()

        with st.container(border=True):
            checked_badge = " ✅" if idx in checked_set else ""
            st.markdown(f"**{idx}. {title}{checked_badge}**")
            if purpose:
                st.markdown(f"- *Purpose:* {purpose}")
            if action:
                st.markdown(f"- *Action:* {action}")

            col1, col2 = st.columns([1, 3])

            if tool == "mark_checked":
                btn_label = "Mark checked"
                runnable = True
            else:
                btn_label = "Run step"
                runnable = tool != "noop"

            with col1:
                # FIX: comma after key=...
                if st.button(
                    btn_label,
                    key=f"run_plan_step_{plan_sig}_{idx}",
                    disabled=not runnable,
                ):
                    df_out, msg = run_plan_step(step, plan_sig=plan_sig, step_idx=idx)
                    st.session_state["plan_last_run"] = {
                        "idx": idx,
                        "msg": msg,
                        "df": df_out.to_dict(orient="records") if isinstance(df_out, pd.DataFrame) else None,
                        "columns": list(df_out.columns) if isinstance(df_out, pd.DataFrame) else None,
                    }

            with col2:
                last = st.session_state.get("plan_last_run", {})
                if last and last.get("idx") == idx:
                    st.success(last.get("msg", "Done."))

                    if last.get("df") is None:
                        st.info("No table output for this step.")
                    else:
                        df_show = pd.DataFrame(last["df"])
                        st.dataframe(df_show, use_container_width=True)

                        # Step2: lane drill-down chart
                        if {"seller_state", "customer_state", "n_late", "late_rate"}.issubset(set(df_show.columns)):
                            st.markdown("**Ops chart: Top lanes**")
                            _render_lane_chart(df_show)

                        # Step3: weekly trend dual-axis chart
                        if "week_start" in df_show.columns:
                            st.markdown("**Ops chart: Weekly trend**")
                            _render_weekly_dual_axis(df_show)


def render_message(msg: Dict):
    with st.chat_message(msg["role"]):
        if msg.get("type") == "df":
            if msg.get("caption"):
                st.markdown(msg["caption"])
            st.dataframe(pd.DataFrame(msg["data"]), use_container_width=True)
        else:
            st.write(msg.get("content", ""))


# =========================================================
# Strong deterministic fallback
# =========================================================
def _fmt_pct(x: float) -> str:
    try:
        return f"{(float(x) * 100.0):.2f}%"
    except Exception:
        return "NA"


def _fmt_num(x, nd: int = 2) -> str:
    try:
        if x is None:
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def _fmt_int(x) -> str:
    try:
        return str(int(x))
    except Exception:
        return "NA"


def _hours_to_days_str(h) -> str:
    try:
        hh = float(h)
        dd = hh / 24.0
        return f"{hh:.1f}h (~{dd:.1f}d)"
    except Exception:
        return f"{h}h"


def fallback_interpretation(intent: str, df: pd.DataFrame, next_title: str) -> str:
    if df is None or df.empty:
        return (
            "**Quick interpretation:**\n"
            "- No rows returned under current parameters. Consider widening the time window or lowering `min_orders` before drawing conclusions [P3].\n"
            f"Next action: {next_title} [P5]\n"
        )

    d = df.copy()
    lines: List[str] = ["**Quick interpretation:**"]

    try:
        if intent == "late_delivery":
            if "n_orders" in d.columns:
                d["n_orders_num"] = pd.to_numeric(d["n_orders"], errors="coerce").fillna(0)
            if "late_rate" in d.columns:
                d["late_rate_num"] = pd.to_numeric(d["late_rate"], errors="coerce").fillna(0)
            if "avg_days_late" in d.columns:
                d["avg_days_late_num"] = pd.to_numeric(d["avg_days_late"], errors="coerce")

            if "n_late" in d.columns:
                d["n_late_num"] = pd.to_numeric(d["n_late"], errors="coerce").fillna(0)
                impact = d.sort_values(["n_late_num", "late_rate_num", "n_orders_num"], ascending=False).iloc[0]
                impact_note = f"`n_late`={_fmt_int(impact.get('n_late'))}"
            else:
                impact = d.iloc[0]
                impact_note = ""

            lines.append(
                f"- Impact leader: `{impact.get('category_en','NA')}` {impact_note}, "
                f"`late_rate`={_fmt_pct(impact.get('late_rate'))}, `n_orders`={_fmt_int(impact.get('n_orders'))}, "
                f"`avg_days_late`={_fmt_num(impact.get('avg_days_late'), 2)} [P1][P3]"
            )
            lines.append(f"Next action: {next_title} [P5]")
            return "\n".join(lines)

        if intent == "stage_clue":
            top = d.head(3)
            for _, r in top.iterrows():
                lines.append(
                    f"- `{r.get('category_en','NA')}`: "
                    f"`purchase→approved`={_hours_to_days_str(r.get('avg_h_purchase_to_approved'))}, "
                    f"`approved→carrier`={_hours_to_days_str(r.get('avg_h_approved_to_carrier'))}, "
                    f"`carrier→customer`={_hours_to_days_str(r.get('avg_h_carrier_to_customer'))} "
                    f"(`n_delivered`={_fmt_int(r.get('n_delivered'))}) [P1][P2]"
                )
            lines.append(f"Next action: {next_title} [P5]")
            return "\n".join(lines)

        # review_risk
        top = d.head(3)
        for _, r in top.iterrows():
            lines.append(
                f"- `{r.get('category_en','NA')}`: `low_rating_rate`={_fmt_pct(r.get('low_rating_rate'))}, "
                f"`avg_review_score`={_fmt_num(r.get('avg_review_score'), 2)}, "
                f"`late_rate`={_fmt_pct(r.get('late_rate'))}, "
                f"`n_orders`={_fmt_int(r.get('n_orders'))} [P4][P1][P3]"
            )
        lines.append(f"Next action: {next_title} [P5]")
        return "\n".join(lines)

    except Exception:
        return (
            "**Quick interpretation:**\n"
            "- Metrics computed from SQL; deterministic interpretation failed unexpectedly.\n"
            f"Next action: {next_title} [P5]\n"
        )


# =========================================================
# LLM anti-cross-talk: JSON -> validate -> repair (Stage4.2)
# =========================================================
def _safe_json_loads(text: str) -> Optional[dict]:
    if not text:
        return None
    raw = str(text).strip()

    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _validate_llm_json(
    obj: dict,
    allowed_categories: List[str],
    allowed_fields: List[str],
    next_title: str,
) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "root_not_object"

    summary = obj.get("summary")
    signals = obj.get("signals")
    hypotheses = obj.get("hypotheses")
    checks = obj.get("checks")
    next_action = obj.get("next_action")

    if not isinstance(summary, str) or not summary.strip():
        return False, "summary_missing"

    if not isinstance(signals, list):
        return False, "signals_not_list"
    if not (3 <= len(signals) <= 6):
        return False, f"signals_len_not_3_to_6:{len(signals)}"

    if not isinstance(hypotheses, list) or not all(isinstance(x, str) and x.strip() for x in hypotheses):
        return False, "hypotheses_invalid"
    if not isinstance(checks, list) or not all(isinstance(x, str) and x.strip() for x in checks):
        return False, "checks_invalid"

    if not isinstance(next_action, str) or not next_action.strip():
        return False, "next_action_missing"
    if not next_action.strip().lower().startswith("next action:"):
        return False, "next_action_must_start_with_next_action"

    cat_set = {str(c).strip() for c in allowed_categories if str(c).strip()}
    field_set = {str(f).strip() for f in allowed_fields if str(f).strip()}
    if not cat_set:
        return False, "allowed_categories_empty"
    if not field_set:
        return False, "allowed_fields_empty"

    # Numbers guard:
    # - summary: NO digits at all
    # - hypotheses/checks: digits are NOT allowed EXCEPT inside Ops KB tags like [S1], [S2], [G1] ...
    num_re = re.compile(r"[\d]")
    ops_tag_re = re.compile(r"\[(?:S|G)\d+\]")  # allowed digit pattern in hypotheses/checks

    if num_re.search(summary):
        return False, "summary_contains_number"

    def _has_illegal_digits(text: str) -> bool:
        if not text:
            return False
        # remove allowed ops tags, then check remaining digits
        cleaned = ops_tag_re.sub("", text)
        return bool(num_re.search(cleaned))

    for i, h in enumerate(hypotheses, 1):
        if _has_illegal_digits(h):
            return False, f"hypothesis_{i}_contains_number"

    for i, c in enumerate(checks, 1):
        if _has_illegal_digits(c):
            return False, f"check_{i}_contains_number"

    # ---------------------------------------------------------
    # 2) Ops KB citation tag rule for hypotheses/checks (SOFT)
    #   Goal:
    #   - Do NOT fail if tag missing / not at end (avoid killing the whole interpretation)
    #   - Still enforce: if tags appear, they must be valid forms like [S1]/[G2]/[P3]
    #   - Optional: require "at least one tagged item" across hypotheses+checks
    # ---------------------------------------------------------
    valid_tag_re = re.compile(r"\[(S|G|P)\d+\]")      # tag anywhere, valid format
    invalid_bracket_tag_re = re.compile(r"\[[^\]]+\]")  # any [...] pattern (for catching weird tags)

    def _contains_only_valid_tags(text: str) -> bool:
        """
        Returns True if every [...] tag in text is a valid [S#]/[G#]/[P#].
        """
        if not text:
            return True
        all_brackets = invalid_bracket_tag_re.findall(text)
        if not all_brackets:
            return True
        # every bracket token must match valid_tag_re
        for tok in all_brackets:
            if not valid_tag_re.fullmatch(tok):
                return False
        return True

    tagged_count = 0

    for i, h in enumerate(hypotheses, 1):
        hh = (h or "").strip()
        # If there are bracket tags, they must be valid.
        if not _contains_only_valid_tags(hh):
            return False, f"hypothesis_{i}_has_invalid_tag"
        if valid_tag_re.search(hh):
            tagged_count += 1

    for i, c in enumerate(checks, 1):
        cc = (c or "").strip()
        if not _contains_only_valid_tags(cc):
            return False, f"check_{i}_has_invalid_tag"
        if valid_tag_re.search(cc):
            tagged_count += 1

    # OPTIONAL SOFT requirement:
    # Encourage at least one tagged hypothesis/check, but DO NOT fail.
    # If you want to enforce it strictly, change to:
    #   return False, "missing_ops_kb_tags"
    if tagged_count == 0:
        # keep this as a non-fatal note for debugging/telemetry
        # (validator returns ok, but you can log this if you want)
        pass

    # ---------------------------------------------------------
    # 3) Validate each signal object (category + fields whitelist)
    # ---------------------------------------------------------
    for idx, s in enumerate(signals, 1):
        if not isinstance(s, dict):
            return False, f"signal_{idx}_not_object"

        cat = s.get("category")
        fields = s.get("fields")
        text = s.get("text")

        if not isinstance(cat, str) or not cat.strip():
            return False, f"signal_{idx}_missing_category"
        cat = cat.strip()
        if cat not in cat_set:
            return False, f"signal_{idx}_category_not_allowed:{cat}"

        if not isinstance(fields, list) or len(fields) == 0:
            return False, f"signal_{idx}_fields_missing"
        for f in fields:
            if not isinstance(f, str) or not f.strip():
                return False, f"signal_{idx}_field_invalid"
            ff = f.strip()
            if ff not in field_set:
                return False, f"signal_{idx}_field_not_allowed:{ff}"

        if not isinstance(text, str) or not text.strip():
            return False, f"signal_{idx}_missing_text"

    return True, "ok"


def _format_bullets_from_structured_json(data: dict, next_title: str) -> str:
    out = ["**Quick interpretation:**"]

    summary = (data.get("summary") or "").strip()
    if summary:
        out.append(f"- **Summary:** {summary}")

    signals = data.get("signals", [])
    if isinstance(signals, list) and signals:
        out.append("- **Signals (evidence):**")
        for s in signals:
            if not isinstance(s, dict):
                continue
            cat = str(s.get("category", "")).strip()
            fields = s.get("fields") if isinstance(s.get("fields"), list) else []
            text = str(s.get("text", "")).strip()
            if not text:
                continue

            cat_h = f"`{cat}`" if cat else ""
            if cat and cat in text:
                text = text.replace(cat, cat_h, 1)
                main = text
            elif cat:
                main = f"{cat_h} — {text}"
            else:
                main = text

            out.append(f"  - {main}")

            clean_fields = [str(x).strip() for x in fields if str(x).strip()]
            if clean_fields:
                f_show = ", ".join([f"`{x}`" for x in clean_fields])
                out.append(f"    - fields: {f_show}")

    hypotheses = data.get("hypotheses", [])
    if isinstance(hypotheses, list) and hypotheses:
        out.append("- **Hypotheses (clues, not root cause):**")
        for h in hypotheses:
            hh = str(h).strip()
            if hh:
                out.append(f"  - {hh}")

    checks = data.get("checks", [])
    if isinstance(checks, list) and checks:
        out.append("- **Checks (what to validate next):**")
        for c in checks:
            cc = str(c).strip()
            if cc:
                out.append(f"  - {cc}")

    out.append(f"Next action: {next_title} [P5]")
    return "\n".join(out)


def llm_interpret_with_repair(
    *,
    intent: str,
    user_text: str,
    preview_df: pd.DataFrame,
    next_title: str,
    policy_ctx: str,
    max_repairs: int = 2,
) -> Tuple[Optional[str], str]:
    if preview_df is None or preview_df.empty:
        return None, "preview_df_empty"
    if "category_en" not in preview_df.columns:
        return None, "preview_missing_category_en"

    allowed_categories = [
        str(x).strip()
        for x in preview_df["category_en"].astype(str).tolist()
        if str(x).strip()
    ]
    allowed_fields = [str(c) for c in preview_df.columns.astype(str).tolist()]
    table_preview = preview_df.to_dict(orient="records")

    base_prompt = """

You are a Fulfillment Ops Analyst writing an ops-style interpretation grounded strictly in the preview SQL rows.

Context:
- You have Policy context (Policy labels like [P1]...[P6]) and Ops KB context (SOP + Guardrails).
- Your job is to write a domain-grade, tool-driven diagnosis memo faithful to the preview SQL rows.

Policy notes:
- If you use a Policy rule/definition, cite it as [P#] in the SAME signal bullet.

Ops KB usage rule (IMPORTANT):
- In "hypotheses" and "checks", prioritize Ops KB guidance FIRST when applicable.
- You SHOULD cite Ops KB tags in hypotheses/checks using [S#] (SOP) and [G#] (Guardrails) if applicable.
- Put [S#]/[G#] at the END of the sentence.
- Do not invent SOP/Guardrail content; paraphrase what is present in the provided context.

Hard constraints (MUST follow):
- Use ONLY numeric values that appear in the preview SQL rows. Never invent numbers.
- You may ONLY mention categories in ALLOWED_CATEGORIES (exact match).
- You may ONLY mention field/metric names in FIELD_WHITELIST (exact match).
- Stage gaps are clues, not root cause [P2].
- If sample size is near threshold, state uncertainty [P3].
- NO SQL code. Output ONLY valid JSON. No markdown fences.

CRITICAL "NO NEW NUMBERS" RULE:
- The JSON fields "summary", "hypotheses", and "checks" MUST NOT contain ANY numbers (digits),
  EXCEPT that hypotheses/checks MAY contain Ops KB tags in the form [S#] or [G#] (digits allowed ONLY inside those tags).
- ONLY the "signals[].text" lines may contain other numbers, and those numbers must come from the preview rows.

Return JSON with EXACT keys (return JSON only; no extra keys):
{{
  "summary": "One sentence. No numbers.",
  "signals": [
    {{"category": "...", "fields": ["..."], "text": "1-2 sentences. May contain numbers from preview rows."}}
  ],
  "hypotheses": ["Clue statements. No numbers except [S#]/[G#] tags.", "Same."],
  "checks": ["Actionable validations. No numbers except [S#]/[G#] tags.", "Same."],
  "next_action": "Next action: {next_title} [P5]"
}}

Content quality guidance:
- summary: impact-first, what stands out and why it matters (NO numbers).
- signals: 3-6 evidence bullets; each must tie to preview fields; cite [P#] here when used.
- hypotheses: 2-3 plausible operational clues (NOT root cause). Prefer Ops KB. End with [S#]/[G#] when used.
- checks: 2-4 next validations the analyst should run (NO numbers except [S#]/[G#] tags). Prefer Ops KB. End with [S#]/[G#] when used.
- next_action: must start with "Next action:" and include NEXT_TITLE verbatim.

User question:
{user_text}

Intent:
{intent}

ALLOWED_CATEGORIES:
{allowed_categories}

FIELD_WHITELIST:
{field_whitelist}

Preview SQL rows (ONLY source of truth):
{table_preview}

Policy + Ops KB context:
{policy_ctx}

NEXT_TITLE (must be used verbatim in next_action):
{next_title}
""".strip()

    def _call_once(prompt_text: str) -> Tuple[Optional[dict], str]:
        try:
            raw = gemini_text(prompt_text)
            obj = _safe_json_loads(raw)
            if obj is None:
                return None, "not_json"
            ok, reason = _validate_llm_json(
                obj=obj,
                allowed_categories=allowed_categories,
                allowed_fields=allowed_fields,
                next_title=next_title,
            )
            if not ok:
                return None, reason
            return obj, "ok"
        except Exception as e:
            if _is_quota_error(str(e)):
                raise
            return None, f"{type(e).__name__}: {e}"

    prompt0 = base_prompt.format(
        user_text=user_text,
        intent=intent,
        allowed_categories=json.dumps(allowed_categories, ensure_ascii=False),
        field_whitelist=", ".join(allowed_fields) if allowed_fields else "(none)",
        table_preview=json.dumps(table_preview, ensure_ascii=False),
        policy_ctx=policy_ctx,
        next_title=next_title,
    )

    obj, reason = _call_once(prompt0)
    if obj is not None:
        return _format_bullets_from_structured_json(obj, next_title), ""

    last_reason = reason
    repair_prompt = prompt0
    for _ in range(max_repairs):
        repair_prompt = (
            repair_prompt
            + "\n\n"
            + f"VALIDATION_ERROR: {last_reason}\nRewrite JSON to comply exactly. Return JSON only."
        )
        obj, reason = _call_once(repair_prompt)
        if obj is not None:
            return _format_bullets_from_structured_json(obj, next_title), ""
        last_reason = reason

    return None, f"llm_failed_after_repairs:{last_reason}"


# =========================================================
# Session state
# =========================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hi! Tell me what you want to check (late delivery by category, stage clues, or review risk).\n\n"
                "Examples:\n"
                "- Top late categories in the last 26 weeks\n"
                "- Is the delay mainly pre-carrier or in-transit?\n"
                "- What review risk signals should I check next?"
            ),
        }
    ]

st.session_state.setdefault("pending_next_sql", None)
st.session_state.setdefault("pending_next_title", None)
st.session_state.setdefault("last_llm_error", "")
st.session_state.setdefault("llm_disabled_until", 0.0)

# plan UI state
st.session_state.setdefault("latest_plan", [])
st.session_state.setdefault("latest_plan_sig", "")
st.session_state.setdefault("plan_checked", {})   # {plan_sig: [idx,...]}
st.session_state.setdefault("plan_last_run", {"idx": None, "msg": "", "df": None, "columns": None})


# =========================================================
# UI header
# =========================================================
st.subheader("Chatbot")
st.caption(f"Gemini model: {GEMINI_MODEL} • Policy RAG: {policy_rag_status}")

c1, c2, c3 = st.columns(3)
quick = None
if c1.button("Quick: Top late categories"):
    quick = "Top late categories in the last 26 weeks"
if c2.button("Quick: Pre-carrier or in-transit?"):
    quick = "Is the delay mainly pre-carrier or in-transit?"
if c3.button("Quick: Review risk checklist"):
    quick = "What review risk signals should I check next?"

# render chat history
for msg in st.session_state["messages"]:
    render_message(msg)

# input
user_text = st.chat_input("Ask a question about fulfillment triage… (English)")
if quick and not user_text:
    user_text = quick

# =========================================================
# Main flow
#   - update latest_plan + plan_sig + checked state
#   - DO NOT render plan here (render once below in fixed panel)
# =========================================================
if user_text:
    # defensive defaults
    st.session_state.setdefault("latest_plan", [])
    st.session_state.setdefault("latest_plan_sig", "")
    st.session_state.setdefault("plan_checked", {})
    st.session_state.setdefault("plan_last_run", {"idx": None, "msg": "", "df": None, "columns": None})
    st.session_state.setdefault("pending_next_title", None)
    st.session_state.setdefault("pending_next_sql", None)
    st.session_state.setdefault("last_llm_error", "")
    st.session_state.setdefault("llm_disabled_until", 0.0)

    # user message
    st.session_state["messages"].append({"role": "user", "content": user_text})
    render_message(st.session_state["messages"][-1])

    # route intent
    intent, mode = route_intent(user_text)

    if intent == "unknown":
        bot = (
            f"I couldn't confidently route your request. (router={mode})\n\n"
            "Try one of these:\n"
            "- late delivery rate / on-time vs late\n"
            "- pre-carrier vs in-transit / stage delays\n"
            "- review score / low ratings / satisfaction risk"
        )
        st.session_state["messages"].append({"role": "assistant", "content": bot})
        render_message(st.session_state["messages"][-1])

    else:
        tool_name, tool_fn = SQL_TOOLS[intent]

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": (
                    f"Got it — running **{tool_name}** with current settings "
                    f"(weeks={weeks}, min_orders={min_orders}, top_k={top_k}). (router={mode})"
                ),
            }
        )
        render_message(st.session_state["messages"][-1])

        # 1) Run SQL
        sql = tool_fn(weeks, min_orders, top_k)
        df = run_query(sql)

        preview_n = min(10, len(df))
        preview_df = df.head(preview_n).copy()

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "type": "df",
                "caption": f"Result preview (top {preview_n}):",
                "data": preview_df.to_dict(orient="records"),
            }
        )
        render_message(st.session_state["messages"][-1])

        # 2) Prepare next_title (used by interpretation)
        next_title, next_sql = build_next_query_sql(intent, df)
        st.session_state["pending_next_title"] = next_title
        st.session_state["pending_next_sql"] = next_sql

        # 2.5) Build / store plan (panel)
        new_plan = build_diagnosis_plan(intent=intent, preview_df=preview_df)
        new_sig = json.dumps(new_plan, ensure_ascii=False, sort_keys=True)

        st.session_state["latest_plan"] = new_plan
        st.session_state["latest_plan_sig"] = new_sig

        # reset outputs (stable structure)
        st.session_state["plan_last_run"] = {"idx": None, "msg": "", "df": None, "columns": None}

        # init checked steps for this plan
        if new_sig not in st.session_state["plan_checked"]:
            st.session_state["plan_checked"][new_sig] = []

        # 3) Interpretation (LLM if available, else deterministic)
        now = time.time()
        cooling = now < float(st.session_state.get("llm_disabled_until", 0.0))

        if client is None or mode != "llm" or cooling:
            if client is None:
                st.session_state["last_llm_error"] = "Gemini client unavailable"
            elif mode != "llm":
                st.session_state["last_llm_error"] = ""
            else:
                remaining = int(float(st.session_state["llm_disabled_until"]) - now)
                st.session_state["last_llm_error"] = f"Cooling down after quota error ({remaining}s remaining)"

            interp = fallback_interpretation(intent, df, next_title)

        else:
            policy_ctx = retrieve_policy_context(f"{intent}\n{user_text}")
            st.session_state["last_llm_error"] = ""

            try:
                llm_text, reason = llm_interpret_with_repair(
                    intent=intent,
                    user_text=user_text,
                    preview_df=preview_df,
                    next_title=next_title,
                    policy_ctx=policy_ctx,
                )
                if llm_text is not None and llm_text.strip():
                    interp = llm_text
                else:
                    st.session_state["last_llm_error"] = reason or "llm_unavailable"
                    interp = fallback_interpretation(intent, df, next_title)

            except Exception as e:
                err_text = f"{type(e).__name__}: {e}"
                st.session_state["last_llm_error"] = err_text
                if _is_quota_error(str(e)):
                    wait_s = _extract_retry_seconds(str(e))
                    st.session_state["llm_disabled_until"] = time.time() + wait_s
                interp = fallback_interpretation(intent, df, next_title)

        st.session_state["messages"].append({"role": "assistant", "content": interp})
        render_message(st.session_state["messages"][-1])

# =========================================================
# Diagnosis plan panel (fixed position)
# =========================================================
st.divider()
render_diagnosis_plan(st.session_state.get("latest_plan", []))

# =========================================================
# Minimal debug (optional)
# =========================================================
with st.expander("Debug: Stage 4 status", expanded=False):
    st.write(f"CWD: {os.getcwd()}")
    st.write(f"GEMINI_API_KEY loaded?: {bool(os.getenv('GEMINI_API_KEY','').strip())}")
    st.write(f"google-genai import ok?: {genai is not None}")
    st.write(f"Gemini client ready?: {client is not None}")
    st.write(f"GEMINI_MODEL: {GEMINI_MODEL}")
    st.write(f"Policy file exists?: {os.path.exists(POLICY_PATH)}")
    st.write(f"Chroma dir: {CHROMA_DIR} (exists={os.path.isdir(CHROMA_DIR)})")
    st.write(f"Policy RAG status: {policy_rag_status}")
    st.write(f"OPS KB file exists?: {os.path.exists(OPS_KB_PATH)}")
    st.write(f"OPS_KB_PATH: {OPS_KB_PATH}")
    st.write(f"Last LLM error: {st.session_state.get('last_llm_error','')}")
    remaining = max(0, int(float(st.session_state.get('llm_disabled_until', 0.0)) - time.time()))
    st.write(f"LLM cooldown remaining (s): {remaining}")
    if not RAG_IMPORT_OK:
        st.write(f"RAG import error: {RAG_IMPORT_ERR}")