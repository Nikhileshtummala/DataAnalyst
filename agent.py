"""Ask-Your-Data Analyst - the "brain" of the agent (no UI code in here).

Flow for every question:
  1. Show the model the columns, sample rows and earlier questions (so follow-ups work)
  2. Model replies with JSON: answerable? SQL? chart? assumptions? alternatives?
  3. We run the SQL (read-only) with DuckDB. If it errors, the model gets one retry.
  4. A SECOND, differently-written query re-computes the answer -> verified / mismatch
  5. Chart with anomaly callouts + a short explanation written from the real result rows
"""
import io
import json
import re
import warnings

import duckdb
import matplotlib

matplotlib.use("Agg")  # draw charts without a window
import matplotlib.pyplot as plt
import numpy as np
import ollama
import pandas as pd
import plotly.express as px

MODEL = "qwen2.5:7b"  # change this if you pull a different model
BLUE, RED, GREEN = "#4C78A8", "#E45756", "#54A24B"


# =====================================================================
# 1. LOADING + DATA QUALITY
# =====================================================================
def _is_text(s):
    return (pd.api.types.is_string_dtype(s)
            and not pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_datetime64_any_dtype(s))


def load_dataframe(source, filename):
    """Read a CSV/Excel file, make SQL-friendly column names, auto-detect date columns."""
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    else:
        df = pd.read_csv(source)

    names, seen = [], {}
    for c in df.columns:
        n = re.sub(r"\W+", "_", str(c).strip()).strip("_").lower() or "column"
        if n[0].isdigit():
            n = "c_" + n
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    df.columns = names

    for c in df.columns:
        if _is_text(df[c]):
            # blank strings -> proper missing values
            df[c] = df[c].where(df[c].astype(str).str.strip() != "", other=np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    df[c] = pd.to_datetime(df[c])  # only succeeds if EVERY value is a date
                except (ValueError, TypeError):
                    pass
    return df


_CURRENCY = re.compile(r"[\s$€£₹¥,%]")


def _to_number(s):
    return pd.to_numeric(s.astype(str).str.replace(_CURRENCY, "", regex=True), errors="coerce")


def _text_number_share(s):
    nn = s.dropna()
    return 0.0 if nn.empty else float(_to_number(nn).notna().mean())


def _date_share(s):
    """(share of distinct values that look like dates in ANY format,
        how many rows do not match the column's main date format)"""
    nn = s.dropna().astype(str)
    if nn.empty:
        return 0.0, 0
    uniq = pd.Series(nn.unique()[:5000])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            share = pd.to_datetime(uniq, format="mixed", errors="coerce").notna().mean()
        except (TypeError, ValueError):
            share = pd.to_datetime(uniq, errors="coerce").notna().mean()
        odd = int(pd.to_datetime(nn.head(20000), errors="coerce").isna().sum())
    return float(share), odd


_NEG_COLS = re.compile(r"(unit|qty|quantity|revenue|sales|price|amount|count|orders|users)")


def check_quality(df):
    """Return a list of {'level': 'warning'|'info', 'text': str, 'fixable': bool}."""
    issues = []
    n = len(df)

    def add(level, text, fixable):
        issues.append({"level": level, "text": text, "fixable": fixable})

    if n == 0:
        return [{"level": "warning", "text": "The file has no rows.", "fixable": False}]

    dups = int(df.duplicated().sum())
    if dups:
        add("warning", f"{dups:,} duplicate rows ({dups / n:.1%} of the data) - they would be double-counted in totals.", True)

    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        if miss:
            if pd.api.types.is_numeric_dtype(s):
                add("info", f"'{c}' has {miss:,} empty cells ({miss / n:.1%}). They are ignored in totals and averages (never guessed).", False)
            elif pd.api.types.is_datetime64_any_dtype(s):
                add("warning", f"'{c}' has {miss:,} rows with no date - they can't be placed in any month.", True)
            else:
                add("warning", f"'{c}' has {miss:,} empty values ({miss / n:.1%}).", True)

        if _is_text(s):
            nn = s.dropna()
            if nn.empty:
                continue
            if _text_number_share(nn) >= 0.8:
                add("warning", f"'{c}' looks like numbers stored as text (e.g. '{nn.iloc[0]}') - sums and averages would fail.", True)
                continue
            share, bad = _date_share(nn)
            if share >= 0.7:
                add("warning", f"'{c}' looks like dates, but {bad:,} values are in a different or unreadable format.", True)
                continue
            if nn.nunique() <= 200:
                key = nn.str.strip().str.lower()
                variants = nn.groupby(key).nunique()
                bad_groups = variants[variants > 1]
                if len(bad_groups):
                    ex = nn[key == bad_groups.index[0]].unique()[:3]
                    add("warning", f"'{c}' spells the same value differently ({' / '.join(repr(v) for v in ex)}) - these would be counted as separate groups.", True)
        elif pd.api.types.is_numeric_dtype(s) and _NEG_COLS.search(c) and (s < 0).any():
            add("info", f"'{c}' has {int((s < 0).sum()):,} negative values (refunds/returns, or data errors?).", False)
    return issues


def clean_dataframe(df):
    """Fix what check_quality flags. Returns (clean_df, list_of_actions_taken)."""
    df = df.copy()
    actions = []
    tidied = []

    for c in list(df.columns):
        if not _is_text(df[c]):
            continue
        s = df[c].str.strip()
        if _text_number_share(s) >= 0.8:
            df[c] = _to_number(s)
            actions.append(f"Converted '{c}' from text to numbers (removed $ , % symbols).")
            continue
        share, _ = _date_share(s)
        if share >= 0.7:
            before_missing = int(s.isna().sum())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    df[c] = pd.to_datetime(s, format="mixed", errors="coerce")
                except (TypeError, ValueError):
                    df[c] = pd.to_datetime(s, errors="coerce")
            unreadable = int(df[c].isna().sum()) - before_missing
            actions.append(f"Converted '{c}' to real dates" + (f" ({unreadable} unreadable values left blank)." if unreadable else "."))
            continue
        nn = s.dropna()
        mapping = {}
        for _, raws in nn.groupby(nn.str.lower()):
            most = raws.value_counts().idxmax()
            for r in raws.unique():
                if r != most:
                    mapping[r] = most
        if mapping:
            s = s.replace(mapping)
        if mapping or not s.equals(df[c]):
            tidied.append(c)
        df[c] = s
    if tidied:
        actions.append(f"Trimmed stray spaces / unified spelling and capitalisation in: {', '.join(tidied)}.")

    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            missing = int(df[c].isna().sum())
            if missing:
                df = df[df[c].notna()]
                actions.append(f"Removed {missing:,} rows with no readable date in '{c}'.")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if before - len(df):
        actions.append(f"Removed {before - len(df):,} duplicate rows.")

    filled = []
    for c in df.columns:
        if _is_text(df[c]) and df[c].isna().any():
            df[c] = df[c].fillna("Unknown")
            filled.append(c)
    if filled:
        actions.append(f"Filled empty text cells with 'Unknown' in: {', '.join(filled)}.")

    num_missing = sum(int(df[c].isna().sum()) for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    if num_missing:
        actions.append(f"Left {num_missing:,} empty number cells blank - they're ignored in totals rather than guessed.")

    return df, actions or ["Nothing to clean - your data already looks good."]


# =====================================================================
# 2. DESCRIBING THE DATA
# =====================================================================
def describe_schema(df):
    lines = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            lines.append(f"- {c} (DATE, from {s.min().date()} to {s.max().date()})")
        elif pd.api.types.is_numeric_dtype(s):
            lines.append(f"- {c} (NUMBER, min {s.min()}, max {s.max()})")
        else:
            uniq = s.dropna().unique()
            shown = uniq if len(uniq) <= 12 else uniq[:8]
            label = "values" if len(uniq) <= 12 else "e.g."
            lines.append(f"- {c} (TEXT, {len(uniq)} distinct; {label}: {', '.join(map(str, shown))})")
    return "\n".join(lines)


def date_column(df):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def latest_date(df):
    c = date_column(df)
    return str(df[c].max().date()) if c and not pd.isna(df[c].max()) else "unknown"


def date_assumptions(question, df):
    """Spell out how we read words like 'last month' (so the founder can check them)."""
    c = date_column(df)
    if c is None or pd.isna(df[c].max()):
        return []
    mx = df[c].max()
    q = question.lower()
    notes = []
    partial = mx.normalize() != (mx + pd.offsets.MonthEnd(0)).normalize()
    month = mx.strftime("%B %Y")
    prev = (mx.replace(day=1) - pd.Timedelta(days=1)).strftime("%B %Y")
    compare = re.search(r"drop|grew|grow|change|increase|decrease|fell|fall|rise|declin|vs|compar", q)

    if re.search(r"\b(last|previous|latest|most recent) month\b", q):
        s = f"I read 'last month' as {month} (the latest month in your data)"
        s += f", compared with {prev}." if compare else "."
        if partial:
            s += f" Note: your data only runs to {mx:%d %b %Y}, so {month} is incomplete."
        notes.append(s)
    elif "this month" in q:
        notes.append(f"I read 'this month' as {month} (the latest month in your data).")
    if "last quarter" in q or "this quarter" in q:
        notes.append(f"I read 'quarter' as Q{(mx.month - 1) // 3 + 1} {mx.year} (the latest quarter in your data).")
    if "last week" in q or "this week" in q:
        notes.append(f"I read 'week' as the 7 days ending {mx:%d %b %Y} (the latest date in your data).")
    if "last year" in q or "this year" in q:
        notes.append(f"I read 'year' as {mx.year} (the latest year in your data).")
    if re.search(r"\b(today|yesterday)\b", q):
        notes.append(f"Your data ends on {mx:%d %b %Y}, so I used that as 'today'.")
    return notes


# =====================================================================
# 3. PROMPTS
# =====================================================================
PROMPT = """You are a careful data analyst. A table named `data` is available in DuckDB.

COLUMNS:
{schema}

SAMPLE ROWS:
{sample}

The latest date in the data is {today}. Treat that as "today".
- "last month" = the most recent calendar month in the data.
- "dropped / grew / declined" = % change compared with the period before it.

Reply with ONE JSON object and nothing else, with exactly these keys:
{
  "answerable": true or false,
  "reason": "if not answerable: what exactly is missing from the data, in one sentence",
  "alternatives": ["if not answerable: 2-3 related questions that CAN be answered with the columns above; otherwise []"],
  "sql": "one DuckDB SELECT query on the table `data`",
  "chart_type": "bar" or "line" or "pie",
  "x": "column name in your query result for the x-axis",
  "y": "numeric column name in your query result for the y-axis",
  "color": "optional column name to split into several series, else empty string",
  "assumptions": "one short sentence on how you interpreted anything ambiguous (e.g. how you defined 'growth'), else empty string"
}

RULES:
- Use ONLY the columns listed above, spelled exactly. Never invent columns.
- If the question cannot be answered with these columns (e.g. it needs profit but there is
  no cost column, or it is about something outside the data), set "answerable": false and
  name the missing thing in "reason", e.g. "There is no cost column, so profit can't be calculated."
  Do NOT guess.
- Follow-ups: if the question refers to an earlier one ("break that down by region", "now only
  for Yoga Mat", "and the month before?"), build on the previous query: keep its measure and
  filters, and apply the change.
- "grew fastest" = largest % change between the first and the latest month in the data,
  unless the person names periods. Say which in "assumptions".
- One SELECT statement only. No semicolons. Add LIMIT 50 unless the query is an aggregate.
- Group by month with: date_trunc('month', some_date_column)
- Give every computed column a short alias (e.g. total_revenue, change_pct). Use NULLIF to avoid dividing by zero.
- Use "line" for trends over time, "bar" for comparing categories, "pie" only for shares of a whole.

EXAMPLE of a good "which X dropped last month" query (adapt the column names):
WITH m AS (SELECT cat_col, date_trunc('month', date_col) AS month, SUM(value_col) AS total FROM data GROUP BY 1, 2),
c AS (SELECT cat_col, month, total, LAG(total) OVER (PARTITION BY cat_col ORDER BY month) AS prev_total FROM m)
SELECT cat_col, ROUND((total - prev_total) * 100.0 / NULLIF(prev_total, 0), 1) AS change_pct
FROM c WHERE month = (SELECT MAX(month) FROM m) ORDER BY change_pct
"""

VERIFY_PROMPT = """You are double-checking another analyst's work. A table named `data` is available in DuckDB.

COLUMNS:
{schema}

The latest date in the data is {today}.

QUESTION: {question}

ORIGINAL QUERY:
{sql}

ORIGINAL RESULT (first rows):
{result}

Write a DIFFERENT DuckDB SELECT query that answers the same question by a different route
(for example subqueries or a self-join instead of window functions, or the other way round).
It must return the same kind of output: the same label column(s) and a numeric column with the same
meaning and the same rounding. Do not copy the original query. One statement, no semicolons.
Reply with JSON only: {"sql": "..."}"""

SUGGEST_PROMPT = """You help startup founders explore their data. A table has these columns:
{schema}

The latest date in the data is {today}.

Write 5 short, specific questions in plain English that a founder would want answered, using ONLY these
columns. Mix them: something that dropped or grew over time, a ranking, a trend, a breakdown.
Never ask for something that needs a column that does not exist.
Reply with JSON only: {"questions": ["...", "..."]}"""


def build_prompt(df):
    return (PROMPT
            .replace("{schema}", describe_schema(df))
            .replace("{sample}", df.head(3).to_csv(index=False))
            .replace("{today}", latest_date(df)))


# =====================================================================
# 4. TALKING TO THE MODEL
# =====================================================================
def _parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def ask_llm(messages):
    """Ask Ollama for a JSON reply. temperature=0 -> same question, same answer."""
    r = ollama.chat(model=MODEL, messages=messages, format="json",
                    options={"temperature": 0, "num_ctx": 4096})
    return _parse_json(r["message"]["content"])


def ask_text(prompt):
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2, "num_ctx": 4096})
    return r["message"]["content"].strip()


def _truthy(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def _norm_plan(plan):
    """Small models sometimes return odd types - make the plan safe to use."""
    if not isinstance(plan, dict):
        raise ValueError("reply was not a JSON object")
    plan["answerable"] = _truthy(plan.get("answerable"))
    for k in ("sql", "reason", "assumptions", "chart_type", "x", "y", "color"):
        v = plan.get(k)
        plan[k] = v.strip() if isinstance(v, str) else ""
    alts = plan.get("alternatives")
    plan["alternatives"] = ([a.strip() for a in alts if isinstance(a, str) and a.strip()][:3]
                            if isinstance(alts, list) else [])
    return plan


def _history_messages(history):
    """Last 3 answered questions, so 'now break that down by region' has something to build on."""
    msgs = []
    done = [h for h in (history or []) if h.get("answerable") and h.get("plan")]
    for h in done[-3:]:
        keep = {k: h["plan"].get(k, "") for k in ("answerable", "sql", "chart_type", "x", "y", "color", "assumptions")}
        msgs.append({"role": "user", "content": h["question"]})
        msgs.append({"role": "assistant", "content": json.dumps(keep)})
    return msgs


# =====================================================================
# 5. SUGGESTED QUESTIONS
# =====================================================================
def fallback_questions(df):
    """Questions built from the column types alone (used if the model's ideas are poor/unavailable)."""
    date = date_column(df)
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in df.columns if _is_text(df[c]) and 1 < df[c].nunique() <= 30]
    pref = ("revenue", "sales", "amount", "total", "units", "quantity", "users", "orders", "price")
    measure = next((c for p in pref for c in nums if p in c), nums[0] if nums else None)
    h = lambda c: c.replace("_", " ")
    qs = []
    if cats and measure and date:
        qs += [f"Which {h(cats[0])} dropped last month?", f"Which {h(cats[0])} grew fastest?"]
    if measure and date:
        qs.append(f"Show total {h(measure)} by month as a trend")
    if cats and measure:
        qs.append(f"What are the top 5 {h(cats[0])} by total {h(measure)}?")
    if len(cats) > 1 and measure:
        qs.append(f"Break down total {h(measure)} by {h(cats[1])}")
    if measure:
        qs.append(f"What is the total {h(measure)}?")
    qs.append("How many rows are in the data?")
    return qs


def suggest_questions(df):
    """5 clickable starter questions: the model's ideas first, topped up with safe fallbacks."""
    qs = []
    try:
        prompt = SUGGEST_PROMPT.replace("{schema}", describe_schema(df)).replace("{today}", latest_date(df))
        r = ask_llm([{"role": "user", "content": prompt}])
        raw = r.get("questions") if isinstance(r, dict) else None
        if isinstance(raw, list):
            for q in raw:
                if isinstance(q, str) and 8 < len(q.strip()) < 140 and q.strip() not in qs:
                    qs.append(q.strip())
    except Exception:
        pass
    for q in fallback_questions(df):
        if len(qs) >= 5:
            break
        if q not in qs:
            qs.append(q)
    return qs[:5]


# =====================================================================
# 6. RUNNING SQL SAFELY
# =====================================================================
BLOCKED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|pragma|export|import|install|load|call)\b", re.I)


def is_safe_sql(sql):
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        return False
    if not re.match(r"(?is)^(select|with)\b", s):
        return False
    return not BLOCKED.search(s)


def run_sql(df, sql):
    con = duckdb.connect()
    con.register("data", df)
    return con.execute(sql.strip().rstrip(";")).df().head(500)


# =====================================================================
# 7. VERIFYING THE ANSWER (second, independent query)
# =====================================================================
def _key(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    k = str(v).strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}", k):
        k += "-01"
    return k[:10] if re.match(r"\d{4}-\d{2}-\d{2}", k) else k


def _close(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= max(0.01 * max(abs(a), abs(b)), 0.06)


def compare_results(a, b, y_a=None):
    """Compare two result tables. Returns (True | False | None, human-readable detail)."""
    num_a = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    num_b = [c for c in b.columns if pd.api.types.is_numeric_dtype(b[c])]
    if not num_a or not num_b:
        return None, "there was no number to compare."
    va = y_a if y_a in num_a else num_a[0]
    vb = va if va in num_b else num_b[0]
    lab_a = [c for c in a.columns if c not in num_a]
    lab_b = [c for c in b.columns if c not in num_b]

    if lab_a and lab_b:
        def rows(df, labs, val):
            # key (order-independent, case-insensitive) -> (number, label to show people)
            return {"|".join(sorted(_key(v) for v in vals)): (num, " / ".join(_label(v) for v in vals))
                    for vals, num in zip(df[labs].itertuples(index=False, name=None), df[val])}
        ra, rb = rows(a.head(50), lab_a, va), rows(b.head(50), lab_b, vb)
        common = set(ra) & set(rb)
        if not common or len(common) < 0.5 * min(len(ra), len(rb)):
            return None, "the two queries returned differently shaped results, so they couldn't be compared."
        bad = [(ra[k][1], ra[k][0], rb[k][0]) for k in sorted(common) if not _close(ra[k][0], rb[k][0])]
        if bad:
            label, x, y = bad[0]
            return False, f"for '{label}' the original gave {x:g} but the second query gave {y:g}."
        return True, f"{len(common)} value(s) matched."
    x, y = a.iloc[0][va], b.iloc[0][vb]
    if _close(x, y):
        return True, f"both gave {x:g}."
    return False, f"the original gave {x:g} but the second query gave {y:g}."


def verify_answer(df, question, sql, result, y_col=None):
    """Re-compute the answer a different way and compare. Never raises."""
    v = {"status": "unverified", "detail": "", "sql": None}
    try:
        prompt = (VERIFY_PROMPT
                  .replace("{schema}", describe_schema(df))
                  .replace("{today}", latest_date(df))
                  .replace("{question}", question)
                  .replace("{sql}", sql)
                  .replace("{result}", result.head(10).to_csv(index=False)))
        vsql = _norm_plan(dict(ask_llm([{"role": "user", "content": prompt}]), answerable=True))["sql"]
        if not vsql or not is_safe_sql(vsql):
            v["detail"] = "the second query wasn't usable."
            return v
        if re.sub(r"\s+", " ", vsql.lower()) == re.sub(r"\s+", " ", sql.lower()):
            v["detail"] = "the second query was identical to the first, so it proves nothing."
            return v
        v["sql"] = vsql
        other = run_sql(df, vsql)
    except Exception as e:
        v["detail"] = f"the second check couldn't run ({str(e).splitlines()[0][:120]})."
        return v

    ok, detail = compare_results(result, other, y_col)
    if ok is True:
        v.update(status="verified", detail=f"a second, independently written query gives the same numbers ({detail})")
    elif ok is False:
        v.update(status="mismatch", detail=f"a second query disagrees: {detail} Check the SQL before trusting this.")
    else:
        v["detail"] = detail
    return v


# =====================================================================
# 8. CHARTS (with anomaly / trend callouts) + PNG export
# =====================================================================
def _resolve_axes(result, spec):
    """Pick x / y / color, double-checking the model's choices against the real result."""
    if result is None or result.empty or len(result.columns) < 2:
        return None
    cols = list(result.columns)
    x = spec.get("x") if spec.get("x") in cols else cols[0]
    numeric = [c for c in cols if c != x and pd.api.types.is_numeric_dtype(result[c])]
    y = spec.get("y") if spec.get("y") in numeric else (numeric[0] if numeric else None)
    if y is None:
        return None
    color = spec.get("color") if spec.get("color") in cols and spec.get("color") not in (x, y) else None
    return x, y, color


def is_chartable(result):
    return _resolve_axes(result, {}) is not None


def _prep(result, x):
    r = result.copy()
    if pd.api.types.is_datetime64_any_dtype(r[x]):
        r = r.sort_values(x)
    return r.reset_index(drop=True)


def _label(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%b %Y") if v.day == 1 else v.strftime("%d %b %Y")
    return str(v)


def _num(v):
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _callouts(r, x, y, color, kind):
    """Find the bar/point that matters: biggest drop, biggest spike, highest, lowest."""
    if color or kind == "table" or len(r) < 2:
        return []
    rr = r[[x, y]].dropna()
    if len(rr) < 2:
        return []
    vals = rr[y].astype(float)
    out = []

    def add(kind_, i, text, short):
        out.append({"kind": kind_, "idx": int(i), "x": r.loc[i, x], "y": float(vals[i]), "text": text, "short": short})

    if kind == "line":
        pct = (vals.pct_change() * 100).replace([np.inf, -np.inf], np.nan)
        if pct.notna().any():
            i = pct.idxmin()
            if pct[i] <= -15:
                add("drop", i, f"▼ Biggest drop: {_label(r.loc[i, x])} ({pct[i]:.0f}% vs the previous point)", f"▼ {pct[i]:.0f}%")
            j = pct.idxmax()
            if pct[j] >= 25:
                add("rise", j, f"▲ Biggest spike: {_label(r.loc[j, x])} (+{pct[j]:.0f}% vs the previous point)", f"▲ +{pct[j]:.0f}%")
    else:
        if (vals < 0).any():
            i = vals.idxmin()
            add("drop", i, f"▼ Biggest decline: {_label(r.loc[i, x])} ({_num(vals[i])})", f"▼ {_num(vals[i])}")
            if (vals > 0).any():
                j = vals.idxmax()
                add("rise", j, f"▲ Biggest gain: {_label(r.loc[j, x])} (+{_num(vals[j])})", f"▲ +{_num(vals[j])}")
        elif len(rr) >= 3:
            j, i = vals.idxmax(), vals.idxmin()
            add("rise", j, f"▲ Highest: {_label(r.loc[j, x])} ({_num(vals[j])})", f"▲ {_num(vals[j])}")
            if i != j:
                add("drop", i, f"▼ Lowest: {_label(r.loc[i, x])} ({_num(vals[i])})", f"▼ {_num(vals[i])}")
    return out


def find_callouts(result, spec, kind=None):
    axes = _resolve_axes(result, spec)
    if axes is None:
        return []
    x, y, color = axes
    kind = (kind or spec.get("chart_type") or "bar").lower()
    return _callouts(_prep(result, x), x, y, color, kind)


def make_chart(result, spec, kind=None):
    """Interactive Plotly chart (or None if the result isn't chartable / kind == 'table')."""
    axes = _resolve_axes(result, spec)
    kind = (kind or spec.get("chart_type") or "bar").lower()
    if axes is None or kind == "table":
        return None
    x, y, color = axes
    r = _prep(result, x)
    if kind == "pie" and (color or (r[y] < 0).any()):
        kind = "bar"

    if kind == "line":
        fig = px.line(r, x=x, y=y, color=color, markers=True)
    elif kind == "pie":
        fig = px.pie(r, names=x, values=y)
    else:
        kind = "bar"
        fig = px.bar(r, x=x, y=y, color=color, barmode="group")

    marks = _callouts(r, x, y, color, kind)
    if kind == "bar" and marks:
        colors = [BLUE] * len(r)
        for c in marks:
            colors[c["idx"]] = RED if c["kind"] == "drop" else GREEN
        fig.update_traces(marker_color=colors)
    if kind in ("bar", "line"):
        for c in marks:
            col = RED if c["kind"] == "drop" else GREEN
            xv = c["x"].isoformat() if isinstance(c["x"], pd.Timestamp) else c["x"]
            if kind == "line":
                fig.add_scatter(x=[xv], y=[c["y"]], mode="markers", marker=dict(size=13, color=col),
                                showlegend=False, hoverinfo="skip")
            fig.add_annotation(x=xv, y=c["y"], text=c["short"], showarrow=True, arrowhead=2, arrowcolor=col,
                               ax=0, ay=-45 if c["y"] >= 0 else 45, font=dict(color=col, size=13),
                               bgcolor="rgba(255,255,255,0.85)", bordercolor=col)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
    return fig


def make_png(result, spec, kind=None):
    """The same chart as a PNG (matplotlib, so no extra browser/kaleido install is needed)."""
    axes = _resolve_axes(result, spec)
    kind = (kind or spec.get("chart_type") or "bar").lower()
    if axes is None or kind == "table":
        return None
    x, y, color = axes
    r = _prep(result, x)
    if kind == "pie" and (color or (r[y] < 0).any()):
        kind = "bar"

    fig, ax = plt.subplots(figsize=(9, 4.8))
    try:
        marks = _callouts(r, x, y, color, kind)
        labels = [_label(v) for v in r[x]]
        if color:
            piv = r.pivot_table(index=x, columns=color, values=y, aggfunc="sum")
            piv.index = [_label(v) for v in piv.index]
            if kind == "line":
                piv.plot(ax=ax, marker="o")
            else:
                piv.plot(kind="bar", ax=ax, width=0.8)
        elif kind == "pie":
            ax.pie(r[y], labels=labels, autopct="%1.0f%%", startangle=90)
            ax.axis("equal")
        elif kind == "line":
            ax.plot(r[x], r[y], marker="o", color=BLUE)
            for c in marks:
                col = RED if c["kind"] == "drop" else GREEN
                ax.scatter([c["x"]], [c["y"]], color=col, s=80, zorder=5)
                ax.annotate(c["short"], xy=(c["x"], c["y"]), xytext=(0, 28 if c["y"] >= 0 else -28),
                            textcoords="offset points", ha="center", color=col, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color=col))
        else:
            pos = list(range(len(r)))
            colors = [BLUE] * len(r)
            for c in marks:
                colors[c["idx"]] = RED if c["kind"] == "drop" else GREEN
            ax.bar(pos, r[y], color=colors)
            ax.set_xticks(pos)
            ax.set_xticklabels(labels)
            ax.margins(y=0.15)  # room for the callout labels above/below the bars
            for c in marks:
                col = RED if c["kind"] == "drop" else GREEN
                ax.annotate(c["short"], xy=(c["idx"], c["y"]), xytext=(0, 14 if c["y"] >= 0 else -20),
                            textcoords="offset points", ha="center", color=col, fontweight="bold")
        if kind != "pie":
            ax.set_xlabel(x.replace("_", " "))
            ax.set_ylabel(y.replace("_", " "))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.25)
            if len(r) > 6 and kind == "bar":
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)


# =====================================================================
# 9. THE MAIN ENTRY POINT
# =====================================================================
def answer_question(df, question, history=None, verify=True):
    out = {"question": question, "answerable": False, "sql": None, "result": None,
           "plan": {}, "chart_spec": {}, "explanation": "", "error": None,
           "assumptions": date_assumptions(question, df), "alternatives": [], "verification": None}

    messages = ([{"role": "system", "content": build_prompt(df)}]
                + _history_messages(history)
                + [{"role": "user", "content": question}])

    def refuse(plan):
        out["explanation"] = plan["reason"] or "This can't be answered from the columns in your data."
        out["alternatives"] = plan["alternatives"] or fallback_questions(df)[:3]
        out["plan"] = plan
        return out

    try:
        plan = _norm_plan(ask_llm(messages))
    except Exception as e:  # broken JSON, or Ollama is down
        out["error"] = f"Could not get a usable reply from the model ({str(e)[:150]})."
        out["explanation"] = out["error"] + " Check that Ollama is running, or try rephrasing."
        out["alternatives"] = fallback_questions(df)[:3]
        return out

    if not plan["answerable"] or not plan["sql"]:
        return refuse(plan)

    # Run the SQL. If it fails, show the model the error and let it fix its query once.
    sql, result, error = plan["sql"], None, None
    for attempt in range(2):
        if not is_safe_sql(sql):
            error = "Only a single read-only SELECT query is allowed."
        else:
            try:
                result = run_sql(df, sql)
                error = None
                break
            except Exception as e:
                error = str(e).splitlines()[0]
        if attempt == 0:
            messages += [{"role": "assistant", "content": json.dumps({"answerable": True, "sql": sql})},
                         {"role": "user", "content":
                          f"That SQL failed with: {error}\nFix it and reply in the same JSON format."}]
            try:
                plan = _norm_plan(ask_llm(messages))
            except Exception:
                break
            if not plan["answerable"] or not plan["sql"]:
                return refuse(plan)
            sql = plan["sql"]

    out["sql"], out["plan"] = sql, plan
    if result is None:
        out["error"] = error
        out["explanation"] = ("I wasn't able to turn that question into a query that runs on your data. "
                              "Try rephrasing it, or use the exact column names.")
        out["alternatives"] = fallback_questions(df)[:3]
        return out

    out["answerable"], out["result"], out["chart_spec"] = True, result, plan

    if plan["assumptions"] and plan["assumptions"].lower() not in " ".join(out["assumptions"]).lower():
        out["assumptions"].append(plan["assumptions"])

    if result.empty:
        out["explanation"] = "The query ran fine but returned no rows - nothing in the data matches this question."
        return out

    if verify:
        out["verification"] = verify_answer(df, question, sql, result, plan.get("y"))

    try:
        context = (" Context: " + " ".join(out["assumptions"])) if out["assumptions"] else ""
        out["explanation"] = ask_text(
            "You are explaining a data result to a busy startup founder.\n"
            f"Question: {question}\n"
            f"SQL that was run: {sql}\n"
            f"Result (CSV):\n{result.head(30).to_csv(index=False)}\n{context}\n\n"
            "Write 2-3 short sentences that directly answer the question. Use ONLY numbers that appear "
            "in the result. No jargon, no mention of SQL."
        )
    except Exception:
        out["explanation"] = "Here is the result (the model couldn't write a summary this time)."
    return out
