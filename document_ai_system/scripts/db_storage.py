"""
Persistence layer (SQLite), 
stores documents, extracted fields, confidence and review status
"""

import sqlite3
from datetime import datetime

DB_PATH = "document_ai.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        processed_at TEXT,
        status TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS extracted_fields (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        field_name TEXT,
        field_value TEXT,
        confidence REAL,
        source TEXT,
        reviewed INTEGER DEFAULT 0,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )
    """)
    conn.commit()
    conn.close()

def create_document(filename):
    """
    Create or retrieve a document row.
    - If filename exists → return existing document_id
    - Else → insert new row
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM documents WHERE filename=?", (filename,))
    row = cur.fetchone()
    if row:
        doc_id = row[0]
    else:
        cur.execute(
            "INSERT INTO documents (filename, processed_at, status) VALUES (?, ?, ?)",
            (filename, datetime.utcnow().isoformat(), "processed")
        )
        doc_id = cur.lastrowid

    conn.commit()
    conn.close()
    return doc_id


def save_fields(document_id, fields, confidences, sources):
    conn = get_connection()
    cur = conn.cursor()

    # remove old fields for this document (avoids duplicates)
    cur.execute("DELETE FROM extracted_fields WHERE document_id=?", (document_id,))

    for field, value in fields.items():
        cur.execute("""
            INSERT INTO extracted_fields
            (document_id, field_name, field_value, confidence, source)
            VALUES (?, ?, ?, ?, ?)
        """, (document_id, field, value, confidences.get(field, 0.0), sources.get(field, "unknown")))

    conn.commit()
    conn.close()


# mark a field as reviewed
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

# load fields for a document
def load_fields(document_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT field_name, field_value, confidence, source, reviewed
        FROM extracted_fields
        WHERE document_id=?
    """, (document_id,))
    rows = cur.fetchall()
    conn.close()
    return rows