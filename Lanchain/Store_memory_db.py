from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
import sqlite3
import json
from datetime import datetime

load_dotenv()

# Create SQLite database
conn = sqlite3.connect("conversation_memory.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_message TEXT,
        bot_response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

print("=== ConversationBufferMemory + SQLite Demo ===\n")

# Function to save to database
def save_to_db(session_id, user_msg, bot_msg):
    cursor.execute("""
        INSERT INTO conversations (session_id, user_message, bot_response)
        VALUES (?, ?, ?)
    """, (session_id, user_msg, bot_msg))
    conn.commit()

# Function to retrieve from database
def get_from_db(session_id):
    cursor.execute("""
        SELECT user_message, bot_response FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    return cursor.fetchall()

Use session
session_id = "7760"

# Turn 1
print("Turn 1:")
user_input = "My name is Vinay"
bot_response = "Nice to meet you Vinay!"
save_to_db(session_id, user_input, bot_response)
print(f"User: {user_input}")
print(f"Bot: {bot_response}")
print(f"✓ Saved to database\n")

# Turn 2
print("Turn 2:")
user_input = "I work at pwc"
bot_response = "PwC is a great company!"
save_to_db(session_id, user_input, bot_response)
print(f"User: {user_input}")
print(f"Bot: {bot_response}")
print(f"✓ Saved to database\n")

# Turn 3
print("Turn 3:")
user_input = "What's my name?"
bot_response = "Your name is Vinay!"
save_to_db(session_id, user_input, bot_response)
print(f"User: {user_input}")
print(f"Bot: {bot_response}")
print(f"✓ Saved to database\n")

# Retrieve all conversations from database
print("=== All Conversations from Database ===")
conversations = get_from_db(session_id)
for i, (user_msg, bot_msg) in enumerate(conversations, 1):
    print(f"Turn {i}:")
    print(f"  User: {user_msg}")
    print(f"  Bot: {bot_msg}\n")

# Show database content
print("=== Raw Database ===")
cursor.execute("SELECT * FROM conversations")
for row in cursor.fetchall():
    print(row)

conn.close()