import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Document AI Review Dashboard", layout="wide")
st.title("Document AI Review Dashboard")

BASE_DIR = Path(__file__).parent
EXCEL_PATH = BASE_DIR / "data" / "processed" / "processed_invoices.xlsx"
LOW_CONF_THRESHOLD = 0.8  # highlight fields below this

@st.cache_data
def load_excel(path):
    """Load Excel with multiple sheets into dict of DataFrames"""
    xls = pd.ExcelFile(path)
    # Combine sheets into one long DataFrame
    data = pd.concat([xls.parse(sheet).assign(field_name=sheet) for sheet in xls.sheet_names])
    data.reset_index(drop=True, inplace=True)  # fix duplicate indices
    return data

df_all = load_excel(EXCEL_PATH)

# Sidebar
doc_options = df_all['filename'].unique()
selected_doc = st.sidebar.selectbox("Select Invoice", ["All"] + list(doc_options))

if selected_doc != "All":
    df_doc = df_all[df_all['filename'] == selected_doc]
else:
    df_doc = df_all.copy()


def highlight_confidence(row):
    """Red for low-confidence, white otherwise"""
    color = "#FFCCCC" if row.get('confidence', 1.0) < LOW_CONF_THRESHOLD else "#FFFFFF"
    return ['background-color: {}'.format(color)] * len(row)

# --- Display all fields per selected document ---
st.subheader(f"Fields for: {selected_doc}" if selected_doc != "All" else "All Documents")
st.write(f"Total records: {len(df_doc)}")
st.dataframe(df_doc[['field_name','field_value','confidence','source']].style.apply(highlight_confidence, axis=1))


st.markdown("---")
st.subheader("Confidence Summary")
avg_conf = df_doc['confidence'].mean()
low_conf_count = (df_doc['confidence'] < LOW_CONF_THRESHOLD).sum()
st.write(f"- Average confidence: {avg_conf:.2f}")
st.write(f"- Fields below threshold ({LOW_CONF_THRESHOLD}): {low_conf_count}")


st.subheader("Low-Confidence Fields")
low_conf_df = df_doc[df_doc['confidence'] < LOW_CONF_THRESHOLD]
st.dataframe(low_conf_df[['field_name','field_value','confidence','source']] if not low_conf_df.empty else "No low-confidence fields!")