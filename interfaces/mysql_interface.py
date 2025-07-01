import mysql.connector
import os

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
    )

def save_message(user_id, text):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (user_id, text) VALUES (%s, %s)",
        (user_id, text)
    )
    conn.commit()
    cursor.close()
    conn.close()

def load_all_messages():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM messages")
    results = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return results

def load_unindexed_messages():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM messages WHERE indexed = FALSE")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

def mark_messages_as_indexed(ids):
    if not ids:
        return
    conn = get_connection()
    cursor = conn.cursor()
    query = "UPDATE messages SET indexed = TRUE WHERE id IN ({})".format(
        ",".join(["%s"] * len(ids))
    )
    cursor.execute(query, ids)
    conn.commit()
    cursor.close()
    conn.close()

