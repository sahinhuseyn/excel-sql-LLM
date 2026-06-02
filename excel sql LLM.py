import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os
import sqlite3
import tempfile
import psycopg2

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Chat with Data", page_icon="📊", layout="wide")
st.title("📊 Chat with your Data")
st.caption("Excel, CSV, .db və ya .sql faylı yüklə, suallarını ver")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat2_messages" not in st.session_state:
    st.session_state.chat2_messages = []
if "pg_messages" not in st.session_state:
    st.session_state.pg_messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "conn" not in st.session_state:
    st.session_state.conn = None
if "tables" not in st.session_state:
    st.session_state.tables = {}
if "pg_conn" not in st.session_state:
    st.session_state.pg_conn = None
if "pg_schema" not in st.session_state:
    st.session_state.pg_schema = None


def data_quality_report(df):
    st.subheader("🔍 Data Quality Report")
    issues_found = False

    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if not null_cols.empty:
        issues_found = True
        with st.expander(f"⚠️ Null dəyərlər ({null_cols.sum()} ədəd)", expanded=True):
            for col, count in null_cols.items():
                pct = round(count / len(df) * 100, 1)
                st.warning(f"**{col}**: {count} null dəyər ({pct}%)")

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        neg = df[df[col] < 0]
        if not neg.empty:
            issues_found = True
            with st.expander(f"🚨 Mənfi dəyərlər — `{col}` ({len(neg)} sətir)", expanded=True):
                st.error(f"**{col}** sütununda {len(neg)} mənfi dəyər var!")
                st.dataframe(neg, use_container_width=True)

    discount_cols = [c for c in df.columns if "discount" in c.lower()]
    for col in discount_cols:
        high = df[df[col] > 100]
        if not high.empty:
            issues_found = True
            with st.expander(f"🚨 Anormal endirim — `{col}` ({len(high)} sətir)", expanded=True):
                st.error(f"**{col}** sütununda 100%-dən yüksək endirim dəyərləri var!")
                st.dataframe(high, use_container_width=True)

    dupes = df.duplicated().sum()
    if dupes > 0:
        issues_found = True
        with st.expander(f"⚠️ Dublikat sətirlər ({dupes} ədəd)", expanded=True):
            st.warning(f"{dupes} dublikat sətir tapıldı.")
            st.dataframe(df[df.duplicated()], use_container_width=True)

    with st.expander("📊 Ümumi statistika", expanded=False):
        st.dataframe(df.describe(include="all"), use_container_width=True)

    if not issues_found:
        st.success("✅ Heç bir problem tapılmadı!")


def load_to_sqlite(tables_dict):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row

    skip = {"sqlite_sequence", "sqlite_master", "sqlite_stat1", "sqlite_stat2"}

    for table_name, df in tables_dict.items():
        if table_name.lower() in skip:
            continue

        clean_table = table_name.replace(" ", "_").replace("-", "_")

        df.columns = [
            str(c).replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]

        df.to_sql(clean_table, conn, if_exists="replace", index=False)

    return conn


def get_schema_for_llm(conn, tables):
    schema_parts = []

    skip = {
        "sqlite_sequence",
        "sqlite_master",
        "sqlite_stat1",
        "sqlite_stat2"
    }

    for table_name in tables:
        if table_name.lower() in skip:
            continue

        clean_table = table_name.replace(" ", "_").replace("-", "_")

        try:
            cursor = conn.execute(f'PRAGMA table_info("{clean_table}")')
            cols = cursor.fetchall()
            col_defs = ", ".join([f"{c[1]} {c[2]}" for c in cols])

            cursor2 = conn.execute(f'SELECT COUNT(*) FROM "{clean_table}"')
            row_count = cursor2.fetchone()[0]

            schema_parts.append(
                f"Table: {clean_table} ({row_count} sətir)\n"
                f"Sütunlar: {col_defs}"
            )

        except Exception as e:
            print(f"Schema xətası ({table_name}): {e}")

    return "\n\n".join(schema_parts)


def text_to_sql(question, schema):
    system_prompt = f"""Sən bir SQL ekspertisən. Sənə verilənlər bazasının sxemi göstərilir.
İstifadəçinin sualına uyğun SQLite SQL sorğusu yaz.

VERİLƏNLƏR BAZASI SXEMİ:
{schema}

QAYDALAR:
- Yalnız SELECT sorğuları yaz (INSERT, UPDATE, DELETE yoxdur)
- SQLite sintaksisindən istifadə et
- Yalnız SQL kodu yaz, heç bir izahat yox
- SQL-i ```sql ``` bloku içinə yaz
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
        max_tokens=512
    )
    raw = response.choices[0].message.content
    if "```sql" in raw:
        sql = raw.split("```sql")[1].split("```")[0].strip()
    elif "```" in raw:
        sql = raw.split("```")[1].split("```")[0].strip()
    else:
        sql = raw.strip()
    return sql


def explain_result(question, sql, result_df):
    result_str = result_df.head(20).to_string(index=False) if not result_df.empty else "Nəticə boşdur."
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Sən bir data analitikisin. SQL sorğusunun nəticəsini Azərbaycan dilində qısa izah et."},
            {"role": "user", "content": f"Sual: {question}\nSQL: {sql}\nNəticə:\n{result_str}"}
        ],
        temperature=0.3,
        max_tokens=256
    )
    return response.choices[0].message.content


def get_pg_schema(pg_conn):
    cursor = pg_conn.cursor()
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """)
    table_names = [row[0] for row in cursor.fetchall()]
    schema_parts = []
    for tname in table_names:
        cursor.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{tname}'
        """)
        cols = cursor.fetchall()
        col_defs = ", ".join([f"{c[0]} {c[1]}" for c in cols])
        cursor.execute(f"SELECT COUNT(*) FROM public.{tname}")
        row_count = cursor.fetchone()[0]
        schema_parts.append(f"Table: {tname} ({row_count} sətir)\nSütunlar: {col_defs}")
    cursor.close()
    return "\n\n".join(schema_parts), table_names


def text_to_pg_sql(question, schema):
    system_prompt = f"""Sən bir SQL ekspertisən. Sənə PostgreSQL verilənlər bazasının sxemi göstərilir.
İstifadəçinin sualına uyğun PostgreSQL SQL sorğusu yaz.

VERİLƏNLƏR BAZASI SXEMİ:
{schema}

QAYDALAR:
- Yalnız SELECT sorğuları yaz (INSERT, UPDATE, DELETE yoxdur)
- PostgreSQL sintaksisindən istifadə et
- public. sxem prefiksini istifadə et (məs: public.employees)
- Yalnız SQL kodu yaz, heç bir izahat yox
- SQL-i ```sql ``` bloku içinə yaz
- Əgər istifadəçi hansı table istifadə ediləcəyini demirsə, ən uyğun table-i seç
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
        max_tokens=512
    )
    raw = response.choices[0].message.content
    if "```sql" in raw:
        sql = raw.split("```sql")[1].split("```")[0].strip()
    elif "```" in raw:
        sql = raw.split("```")[1].split("```")[0].strip()
    else:
        sql = raw.strip()
    return sql


# --- Sidebar ---
with st.sidebar:
    st.header("📁 Fayl yüklə")
    st.caption("Excel, CSV, .db və ya .sql faylı yüklə")

    uploaded_files = st.file_uploader(
        "Fayl seç",
        type=["csv", "xlsx", "xls", "db", "sqlite", "sqlite3", "sql"],
        accept_multiple_files=True
    )

    if uploaded_files:
        tables = {}
        skip = {"sqlite_sequence", "sqlite_master", "sqlite_stat1", "sqlite_stat2"}

        for f in uploaded_files:
            if f.name.endswith(".csv"):
                df = pd.read_csv(f)
                table_name = os.path.splitext(f.name)[0].replace(" ", "_").replace("-", "_")
                tables[table_name] = df
                st.success(f"✅ {f.name} → `{table_name}`")
                st.caption(f"{df.shape[0]} sətir, {df.shape[1]} sütun")

            elif f.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(f)
                table_name = os.path.splitext(f.name)[0].replace(" ", "_").replace("-", "_")
                tables[table_name] = df
                st.success(f"✅ {f.name} → `{table_name}`")
                st.caption(f"{df.shape[0]} sətir, {df.shape[1]} sütun")

            elif f.name.endswith((".db", ".sqlite", ".sqlite3")):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                db_conn = sqlite3.connect(tmp_path, check_same_thread=False)
                cursor = db_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                db_tables = [row[0] for row in cursor.fetchall()]
                for tname in db_tables:
                    if tname.lower() in skip:
                        continue
                    df = pd.read_sql_query(f"SELECT * FROM {tname}", db_conn)
                    tables[tname] = df
                    st.success(f"✅ {f.name} → `{tname}`")
                    st.caption(f"{df.shape[0]} sətir, {df.shape[1]} sütun")
                db_conn.close()
                os.unlink(tmp_path)

            elif f.name.endswith(".sql"):
                import re as _re
                sql_content = f.read().decode("utf-8", errors="ignore")

                bad_patterns = [
                    "SET ",
                    "ENGINE=",
                    "AUTO_INCREMENT",
                    "LOCK TABLES",
                    "UNLOCK TABLES",
                    "DELIMITER",
                    "/*!",
                    "COMMENT ON",
                    "CREATE SEQUENCE",
                    "ALTER SEQUENCE",
                    "DROP SEQUENCE",
                    "NEXTVAL",
                    "CURRVAL",
                    "SETVAL",
                    "CREATE INDEX",
                    "CREATE UNIQUE INDEX",
                    "CREATE OR REPLACE",
                    "CREATE SCHEMA",
                    "SET search_path",
                    "GRANT ",
                    "REVOKE ",
                    "OWNER TO",
                    "RETURNING ",
                    "ON CONFLICT",
                ]

                cleaned_lines = []
                for line in sql_content.splitlines():
                    stripped = line.strip()
                    upper = stripped.upper()
                    if any(bad.upper() in upper for bad in bad_patterns):
                        continue
                    if stripped.startswith("--") or stripped == "":
                        continue
                    cleaned_lines.append(line)

                sql_content = "\n".join(cleaned_lines)
                # Bitişik CREATE TABLE-lar arasında nöqtəli vergül çatışmırsa əlavə et
                sql_content = _re.sub(r'\)\s*\n\s*CREATE', ');\nCREATE', sql_content, flags=_re.IGNORECASE)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                    tmp_path = tmp.name
                db_conn = sqlite3.connect(tmp_path, check_same_thread=False)
                try:
                    db_conn.executescript(sql_content)
                    db_conn.commit()
                    cursor = db_conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                    db_tables = [row[0] for row in cursor.fetchall()]
                    for tname in db_tables:
                        if tname.lower() in skip:
                            continue
                        df = pd.read_sql_query(f"SELECT * FROM {tname}", db_conn)
                        tables[tname] = df
                        st.success(f"✅ {f.name} → `{tname}`")
                        st.caption(f"{df.shape[0]} sətir, {df.shape[1]} sütun")
                except Exception as e:
                    st.error(f"SQL faylı xətası: {e}")
                    # Problem yaradan konkret ifadəni tap
                    for i, stmt in enumerate(sql_content.split(";")):
                        stmt = stmt.strip()
                        if not stmt:
                            continue
                        try:
                            db_conn.execute(stmt)
                        except Exception as stmt_err:
                            st.warning(f"⚠️ Problem ({i+1}): `{stmt[:100]}`\nXəta: {stmt_err}")
                finally:
                    db_conn.close()
                    os.unlink(tmp_path)

        if tables:
            st.session_state.tables = tables
            st.session_state.df = list(tables.values())[0]
            st.session_state.conn = load_to_sqlite(tables)
            st.session_state.messages = []
            st.session_state.chat2_messages = []

            st.divider()
            st.subheader("📋 Cədvəllər")
            for tname, tdf in tables.items():
                with st.expander(f"`{tname}`"):
                    schema_df = pd.DataFrame({
                        "Sütun": tdf.columns,
                        "Tip": tdf.dtypes.astype(str).values,
                        "Null": tdf.isnull().sum().values
                    })
                    st.dataframe(schema_df, hide_index=True, use_container_width=True)


# --- Ana hissə: Tab-lar həmişə görünür ---
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat (SQL)", "📊 Chat (Data)", "🔍 Data Quality", "🐘 PostgreSQL"])

FORBIDDEN_WORDS = ["drop", "delete", "update", "insert", "alter", "truncate", "create", "replace"]

# Tab 1: Text-to-SQL
with tab1:
    if st.session_state.conn is not None:
        conn = st.session_state.conn
        tables = st.session_state.tables
        schema = get_schema_for_llm(conn, list(tables.keys()))
        st.caption("Azərbaycan dilində sual ver — AI SQL yazıb nəticəni göstərəcək")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("sql"):
                    with st.expander("🔍 Yazılan SQL"):
                        st.code(msg["sql"], language="sql")
                if msg.get("table") is not None:
                    st.dataframe(msg["table"], use_container_width=True)
                st.markdown(msg["content"])

        if question := st.chat_input("Sual ver... (məs: IT departamentində neçə işçi var?)"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("SQL yazılır..."):
                    sql = text_to_sql(question, schema)
                with st.expander("🔍 Yazılan SQL", expanded=True):
                    st.code(sql, language="sql")
                try:
                    if any(word in sql.lower() for word in FORBIDDEN_WORDS):
                        st.error("❌ Təhlükəli SQL bloklandı!")
                        st.stop()
                    result_df = pd.read_sql_query(sql, conn)
                    st.dataframe(result_df, use_container_width=True)
                    with st.spinner("İzah edilir..."):
                        explanation = explain_result(question, sql, result_df)
                    st.markdown(explanation)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": explanation,
                        "sql": sql,
                        "table": result_df
                    })
                except Exception as e:
                    err_msg = f"SQL xətası: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
    else:
        st.info("⬅️ Sol tərəfdən fayl yüklə")

# Tab 2: Data Chat
with tab2:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.caption("Data haqqında ümumi suallar ver")

        with st.expander("📋 Data preview (ilk 5 sətir)"):
            st.dataframe(df.head(), use_container_width=True)

        for msg in st.session_state.chat2_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question2 := st.chat_input("Sual ver...", key="chat2"):
            st.session_state.chat2_messages.append({"role": "user", "content": question2})
            with st.chat_message("user"):
                st.markdown(question2)

            sample = df.head(50).to_string(index=False)
            stats = df.describe(include="all").to_string()
            schema_info = "\n".join([f"- {c}: {t}" for c, t in zip(df.columns, df.dtypes)])
            system_prompt = f"""Sən data analitikisin. Azərbaycan dilində cavab ver.
Sətir sayı: {len(df)}, Sütun sayı: {len(df.columns)}
Schema:\n{schema_info}
Statistika:\n{stats}
İlk 50 sətir:\n{sample}"""

            with st.chat_message("assistant"):
                with st.spinner("Düşünür..."):
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *[{"role": m["role"], "content": m["content"]}
                              for m in st.session_state.chat2_messages]
                        ],
                        temperature=0.3,
                        max_tokens=1024
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)

            st.session_state.chat2_messages.append({"role": "assistant", "content": answer})
    else:
        st.info("⬅️ Sol tərəfdən fayl yüklə")

# Tab 3: Data Quality
with tab3:
    if st.session_state.df is not None:
        data_quality_report(st.session_state.df)
    else:
        st.info("⬅️ Sol tərəfdən fayl yüklə")

# Tab 4: PostgreSQL
with tab4:
    st.caption("PostgreSQL verilənlər bazasına qoşul və suallarını ver")

    with st.expander("🔌 Qoşulma məlumatları", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            pg_host = st.text_input("Host", value="localhost")
            pg_db = st.text_input("Database", value="postgres")
            pg_user = st.text_input("User", value="postgres")
        with col2:
            pg_port = st.text_input("Port", value="5432")
            pg_pass = st.text_input("Password", type="password")
        connect_btn = st.button("🔌 Qoşul")

    if connect_btn:
        try:
            pg_conn = psycopg2.connect(
                host=pg_host, port=pg_port,
                database=pg_db, user=pg_user, password=pg_pass
            )
            st.session_state.pg_conn = pg_conn
            st.session_state.pg_messages = []
            pg_schema, pg_tables = get_pg_schema(pg_conn)
            st.session_state.pg_schema = pg_schema
            st.session_state.pg_tables = pg_tables
            st.success(f"✅ Qoşuldu! {len(pg_tables)} cədvəl tapıldı: {', '.join(pg_tables)}")
        except Exception as e:
            st.error(f"Qoşulma xətası: {e}")

    if st.session_state.pg_conn is not None:
        st.divider()

        with st.expander("📋 Verilənlər bazası sxemi"):
            st.code(st.session_state.pg_schema)

        for msg in st.session_state.pg_messages:
            with st.chat_message(msg["role"]):
                if msg.get("sql"):
                    with st.expander("🔍 Yazılan SQL"):
                        st.code(msg["sql"], language="sql")
                if msg.get("table") is not None:
                    st.dataframe(msg["table"], use_container_width=True)
                st.markdown(msg["content"])

        if pg_question := st.chat_input("PostgreSQL sualını ver...", key="pg_chat"):
            st.session_state.pg_messages.append({"role": "user", "content": pg_question})
            with st.chat_message("user"):
                st.markdown(pg_question)
            with st.chat_message("assistant"):
                with st.spinner("SQL yazılır..."):
                    sql = text_to_pg_sql(pg_question, st.session_state.pg_schema)
                with st.expander("🔍 Yazılan SQL", expanded=True):
                    st.code(sql, language="sql")
                try:
                    if any(word in sql.lower() for word in FORBIDDEN_WORDS):
                        st.error("❌ Təhlükəli SQL bloklandı!")
                        st.stop()
                    result_df = pd.read_sql_query(sql, st.session_state.pg_conn)
                    st.dataframe(result_df, use_container_width=True)
                    with st.spinner("İzah edilir..."):
                        explanation = explain_result(pg_question, sql, result_df)
                    st.markdown(explanation)
                    st.session_state.pg_messages.append({
                        "role": "assistant",
                        "content": explanation,
                        "sql": sql,
                        "table": result_df
                    })
                except Exception as e:
                    err_msg = f"SQL xətası: {e}"
                    st.error(err_msg)
                    st.session_state.pg_messages.append({"role": "assistant", "content": err_msg})
