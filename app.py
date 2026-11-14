"""Streamlit UI for the Ask-Your-Data Analyst.   Run with:  streamlit run app.py"""
import os

import ollama
import streamlit as st

import agent

st.set_page_config(page_title="Ask-Your-Data Analyst", page_icon="📊", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "sample_sales.csv")

st.title("📊 Ask-Your-Data Analyst")
st.caption(f"Upload a spreadsheet, ask questions in plain English. Runs 100% locally on {agent.MODEL}.")

# ---------------------------------------------------------------- is Ollama ready?
try:
    listing = ollama.list()
except Exception:
    st.error("Can't reach Ollama. Open the Ollama app (or run `ollama serve`) and refresh this page.")
    st.stop()
try:
    names = [m["model"] for m in listing["models"]]
except Exception:
    names = []
if names and not any(n == agent.MODEL or n.startswith(agent.MODEL.split(":")[0] + ":") for n in names):
    st.error(f"Model not found. Run:  ollama pull {agent.MODEL}")
    st.stop()

# ---------------------------------------------------------------- session state
for key, default in {"history": [], "pending": None, "df": None, "file_name": None, "upload_key": None,
                     "suggestions": [], "quality": [], "clean_actions": []}.items():
    st.session_state.setdefault(key, default)


def safe(text):
    """Streamlit treats $...$ as maths - escape dollar signs so money shows up normally."""
    return (text or "").replace("$", "\\$")


def queue(question):
    st.session_state.pending = question


def load_new(df, name):
    st.session_state.update(df=df, file_name=name, history=[], clean_actions=[], pending=None)
    st.session_state.quality = agent.check_quality(df)
    with st.spinner("Reading your data and thinking of good questions..."):
        st.session_state.suggestions = agent.suggest_questions(df)


def do_clean():
    df, actions = agent.clean_dataframe(st.session_state.df)
    st.session_state.update(df=df, clean_actions=actions, history=[])
    st.session_state.quality = agent.check_quality(df)


def clear_chat():
    st.session_state.history = []


# ---------------------------------------------------------------- sidebar: data
with st.sidebar:
    st.header("1. Your data")
    uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    use_sample = st.button("Use sample sales data", disabled=not os.path.exists(SAMPLE))

try:
    if uploaded is not None and st.session_state.upload_key != (uploaded.name, uploaded.size):
        st.session_state.upload_key = (uploaded.name, uploaded.size)
        load_new(agent.load_dataframe(uploaded, uploaded.name), uploaded.name)
    elif use_sample:
        load_new(agent.load_dataframe(SAMPLE, "sample_sales.csv"), "sample_sales.csv")
except Exception as e:
    st.error(f"Couldn't read that file: {e}")
    st.stop()

if st.session_state.df is None:
    st.info("👈 Upload a CSV or Excel file (or click **Use sample sales data**) to get started.")
    st.stop()

df = st.session_state.df

with st.sidebar:
    st.success(f"{st.session_state.file_name}: {len(df):,} rows, {len(df.columns)} columns")
    with st.expander("What the agent sees"):
        st.text(agent.describe_schema(df))

    st.header("2. Settings")
    verify = st.toggle("Double-check answers (slower, more trustworthy)", value=True)

    st.header("🕘 Question history")
    latest = {}
    for item in st.session_state.history:
        latest[item["question"]] = item
    asked = list(reversed(list(latest)))
    if not asked:
        st.caption("Your questions will show up here. Click one to run it again.")
    for j, q in enumerate(asked[:12]):
        icon = "✅" if latest[q]["answerable"] else "⛔"
        st.button(f"{icon} {q}", key=f"hist{j}", on_click=queue, args=(q,), help="Run this question again", width="stretch")
    if st.session_state.history:
        st.button("🗑 Clear chat", on_click=clear_chat)

# ---------------------------------------------------------------- data quality
issues = st.session_state.quality
fixable = any(i["fixable"] for i in issues)
label = (f"🧹 Data quality check - {len(issues)} thing(s) worth knowing" if issues
         else "🧹 Data quality check - your data looks clean ✅")
with st.expander(label, expanded=fixable):
    for i in issues:
        (st.warning if i["level"] == "warning" else st.info)(safe(i["text"]))
    if fixable:
        st.button("✨ Auto-clean my data", on_click=do_clean, type="primary")
        st.caption("Cleaning never invents numbers. Every change is listed afterwards.")
    if st.session_state.clean_actions:
        st.markdown("**What I changed:**")
        for a in st.session_state.clean_actions:
            st.success(safe(a))

with st.expander("Preview your data"):
    st.dataframe(df.head(20), width="stretch")

# ---------------------------------------------------------------- suggested questions
if st.session_state.suggestions:
    with st.expander("💡 Suggested questions (click one)", expanded=not st.session_state.history):
        cols = st.columns(2)
        for j, q in enumerate(st.session_state.suggestions):
            cols[j % 2].button(q, key=f"sug{j}", on_click=queue, args=(q,), width="stretch")


# ---------------------------------------------------------------- one answer
def render_item(i, item):
    with st.chat_message("user"):
        st.write(safe(item["question"]))
    with st.chat_message("assistant"):
        # --- can't answer: say why + offer things we CAN answer
        if not item["answerable"]:
            st.warning(safe(item["explanation"]))
            if item["alternatives"]:
                st.markdown("**Here's what I *can* answer instead:**")
                for j, q in enumerate(item["alternatives"]):
                    st.button(q, key=f"alt{i}_{j}", on_click=queue, args=(q,))
            if item["sql"]:
                with st.expander("🔍 The query I tried"):
                    st.code(item["sql"], language="sql")
                    if item["error"]:
                        st.caption(f"Technical detail: {item['error']}")
            return

        st.markdown(safe(item["explanation"]))

        # --- verification badge
        v = item.get("verification")
        if v:
            if v["status"] == "verified":
                st.success("✓ Verified - " + safe(v["detail"]))
            elif v["status"] == "mismatch":
                st.warning("⚠ Double-check this one - " + safe(v["detail"]))
            else:
                st.caption("ℹ️ Couldn't double-check this answer: " + safe(v["detail"]))

        # --- assumptions
        if item["assumptions"]:
            st.info("📝 **Assumptions I made**\n\n" + "\n".join(f"- {safe(a)}" for a in item["assumptions"]))

        result, spec = item["result"], item["chart_spec"]
        chartable = agent.is_chartable(result)
        kind = "table"

        # --- chart (with type switcher) or table
        if chartable:
            kinds = ["Bar", "Line", "Pie", "Table"]
            default = (spec.get("chart_type") or "bar").capitalize()
            kind = st.selectbox("View as", kinds, index=kinds.index(default) if default in kinds else 0,
                                key=f"ct{i}").lower()
        fig = agent.make_chart(result, spec, kind) if chartable else None
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key=f"fig{i}_{kind}")
            for c in agent.find_callouts(result, spec, kind):
                st.caption(safe(c["text"]))
        else:
            st.dataframe(result, width="stretch")

        # --- downloads
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("⬇ Result (CSV)", result.to_csv(index=False).encode(), file_name="result.csv",
                           mime="text/csv", key=f"csv{i}")
        if fig is not None:
            pngs = item.setdefault("png", {})
            if kind not in pngs:
                try:
                    pngs[kind] = agent.make_png(result, spec, kind)
                except Exception:
                    pngs[kind] = None
            if pngs[kind]:
                c2.download_button("⬇ Chart (PNG)", pngs[kind], file_name="chart.png", mime="image/png",
                                   key=f"png{i}_{kind}")

        # --- show the work
        with st.expander("🔍 Show the query I ran (so you can check my work)"):
            st.code(item["sql"], language="sql")
            if v and v.get("sql"):
                st.caption("Second, independently written query used for verification:")
                st.code(v["sql"], language="sql")
            if fig is not None:
                st.dataframe(result, width="stretch")


for i, item in enumerate(st.session_state.history):
    render_item(i, item)

# ---------------------------------------------------------------- new question
typed = st.chat_input("Ask a question about your data... (follow-ups like 'now break that down by region' work too)")
question = typed or st.session_state.pending
st.session_state.pending = None
if question:
    wait = "Thinking... a 7B model on a laptop can take 30-90 seconds" + (" (it also double-checks itself)." if verify else ".")
    with st.spinner(wait):
        out = agent.answer_question(df, question, history=st.session_state.history, verify=verify)
    st.session_state.history.append(out)
    st.rerun()
