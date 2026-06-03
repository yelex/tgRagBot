import mysql.connector
import os
from typing import Optional

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


# ── Orders ──────────────────────────────────────────────────────────────────

_ORDER_FIELDS = {
    "bouquet_name", "bouquet_price", "delivery_address", "delivery_date",
    "delivery_time", "recipient_phone", "card_text", "delivery_cost",
    "total_cost", "payment_status", "order_status", "operator_notes",
}


def create_order(user_id: int, user_name: Optional[str], data: dict) -> int:
    bouquet_price = data.get("bouquet_price") or 0
    delivery_cost = data.get("delivery_cost") or 0
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO orders
        (user_id, user_name, bouquet_name, bouquet_price, delivery_address,
         delivery_date, delivery_time, recipient_phone, card_text,
         delivery_cost, total_cost)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            user_id,
            user_name,
            data.get("bouquet_name"),
            bouquet_price if bouquet_price else None,
            data.get("address"),
            data.get("delivery_date"),
            data.get("delivery_time"),
            data.get("recipient_phone"),
            data.get("card_text"),
            delivery_cost,
            bouquet_price + delivery_cost,
        ),
    )
    conn.commit()
    order_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return order_id


def update_order(order_id: int, **kwargs) -> None:
    fields = {k: v for k, v in kwargs.items() if k in _ORDER_FIELDS}
    if not fields:
        return
    conn = get_connection()
    cursor = conn.cursor()
    set_clause = ", ".join(f"`{k}` = %s" for k in fields)
    cursor.execute(
        f"UPDATE orders SET {set_clause} WHERE id = %s",
        list(fields.values()) + [order_id],
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_active_order(user_id: int) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT * FROM orders WHERE user_id = %s
        AND order_status NOT IN ('closed', 'cancelled')
        ORDER BY created_at DESC LIMIT 1""",
        (user_id,),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row


def get_order(order_id: int) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM orders WHERE id = %s", (order_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row

