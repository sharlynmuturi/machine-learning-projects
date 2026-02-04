import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path

st.set_page_config(page_title="Document AI Review Dashboard", layout="wide")
st.title("Document AI Review Dashboard")

EXCEL_PATH = Path("data/processed/processed_invoices.xlsx")
RAW_DATA_DIR = Path("data/raw") 
LOW_CONF_THRESHOLD = 0.8  # highlight fields below this

@st.cache_data
def load_excel(path):
    """Load Excel with multiple sheets into a dict of DataFrames"""
    xls = pd.ExcelFile(path)
    data = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
    return data

excel_data = load_excel(EXCEL_PATH)

# Sidebar
# select field
field_options = list(excel_data.keys())
selected_field = st.sidebar.selectbox("Select Field (Sheet)", field_options)

df_field = excel_data[selected_field]

# filter by document
doc_options = df_field['filename'].unique()
selected_doc = st.sidebar.selectbox("Filter by Invoice (optional)", ["All"] + list(doc_options))
if selected_doc != "All":
    df_field = df_field[df_field['filename'] == selected_doc]

# Display Table
def highlight_confidence(row):
    """Highlight low-confidence fields red"""
    color = "#FFFFFF"
    if row.get('confidence', 1.0) < LOW_CONF_THRESHOLD:
        color = "#FFCCCC"  # light red
    return ['background-color: {}'.format(color)] * len(row)

st.subheader(f"Field: {selected_field}")
st.write(f"Total records: {len(df_field)}")
st.dataframe(df_field.style.apply(highlight_confidence, axis=1))

# Display raw image if document selected
if selected_doc != "All":
    file_path = RAW_DATA_DIR / selected_doc
    if file_path.exists():
        st.subheader(f"Raw Invoice Preview: {selected_doc}")

        try:
            if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff"]:
                # Image files
                image = Image.open(file_path)
                st.image(image, use_column_width=True)

            elif file_path.suffix.lower() == ".pdf":
                # PDF: convert first page to image
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
