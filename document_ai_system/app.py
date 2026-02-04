import streamlit as st
import sqlite3
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path

DB_PATH = "document_ai.db"
RAW_DATA_DIR = Path("data/raw")
LOW_CONFIDENCE_THRESHOLD = 0.8  # highlight fields below this

# SQLite helper functions
def get_connection():
    return sqlite3.connect(DB_PATH)

def load_documents():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, filename, processed_at, status FROM documents ORDER BY processed_at DESC")
    docs = cur.fetchall()
    conn.close()
    return docs

def load_fields(document_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT field_name, field_value, confidence, source, reviewed
        FROM extracted_fields
        WHERE document_id=?
    """, (document_id,))
    fields = cur.fetchall()
    conn.close()
    return fields

def mark_field_reviewed(document_id, field_name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE extracted_fields
        SET reviewed=1
        WHERE document_id=? AND field_name=?
    """, (document_id, field_name))
    conn.commit()
    conn.close()


st.set_page_config(page_title="Document AI Review Dashboard", layout="wide")
st.title("Document AI Review Dashboard")

# Sidebar
# select document
docs = load_documents()
if not docs:
    st.warning("No documents found in the database.")
    st.stop()

doc_options = {f"{d[1]} ({d[2]})": d[0] for d in docs}  # display name -> id
selected_doc = st.sidebar.selectbox("Select Invoice", options=list(doc_options.keys()))

if selected_doc:
    document_id = doc_options[selected_doc]
    st.header(f"Fields for: {selected_doc}")

    # Load extracted fields
    fields = load_fields(document_id)
    if not fields:
        st.warning("No extracted fields found.")
    else:
        # Layout: field | value | confidence | source | review
        for field_name, field_value, confidence, source, reviewed in fields:
            # color coding
            if confidence < LOW_CONFIDENCE_THRESHOLD and not reviewed:
                color = "#FFDDDD"  # low-confidence red
            else:
                color = "#DDFFDD" if reviewed else "#FFFFFF"  # reviewed green, unreviewed white

            col1, col2, col3, col4, col5 = st.columns([2, 4, 2, 2, 1])
            with col1:
                st.markdown(f"**{field_name}**", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='background-color:{color}; padding:4px'>{field_value}</div>", unsafe_allow_html=True)
            with col3:
                st.write(f"{confidence:.2f}")
            with col4:
                st.write(source)
            with col5:
                if not reviewed:
                    key = f"review_{document_id}_{field_name}"
                    if st.button("Review", key=key):
                        mark_field_reviewed(document_id, field_name)
                        st.experimental_set_query_params(reload=str(document_id))
                        st.success(f"Field '{field_name}' marked as reviewed")

        st.info("Red = low-confidence, White = unreviewed, Green = reviewed")

    # Display raw invoice
    file_path = RAW_DATA_DIR / selected_doc.split(" ")[0]  # strip timestamp from label
    if file_path.exists():
        st.subheader("Raw Invoice Preview")
        try:
            if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff"]:
                image = Image.open(file_path)
                st.image(image, use_column_width=True)
            elif file_path.suffix.lower() == ".pdf":
                pages = convert_from_path(file_path, dpi=200, first_page=1, last_page=1)
                if pages:
                    st.image(pages[0], use_column_width=True)
                else:
                    st.warning("PDF has no pages to display.")
            else:
                st.warning("Unsupported file type for preview.")
        except Exception as e:
            st.error(f"Cannot display file: {e}")
    else:
        st.warning(f"Raw file not found in {RAW_DATA_DIR}")
