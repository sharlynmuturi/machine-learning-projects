from db_storage import get_connection

def reset_db():
    conn = get_connection()
    cur = conn.cursor()

    # Delete all extracted fields first
    cur.execute("DELETE FROM extracted_fields")

    # Delete all documents
    cur.execute("DELETE FROM documents")

    conn.commit()
    conn.close()
    print("Database cleared. Ready for fresh processing!")

if __name__ == "__main__":
    reset_db()
