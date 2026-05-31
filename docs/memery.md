# Task: Implement Finabot Memory Module
Create file: finabot/memory.py
Build a lightweight local memory system for a finance AI agent, no remote services required.

## 1. Directory & Storage Structure
Project relative paths:
- memory/short_term/       : JSON files for session chat history
- memory/working_memory/   : JSON files for agent task & reasoning data
- memory/long_term.db      : SQLite database for user permanent memory
- memory/knowledge/        : Persistent Chroma vector database for finance knowledge

Auto create all missing folders during initialization.

## 2. Module Details & Functions
### 2.1 Short-Term Memory (Conversation History)
- Persist per session_id as .json file
- Functions:
  def save_short_memory(session_id: str, messages: list) -> None
  def load_short_memory(session_id: str) -> list
- File rule: UTF-8, json indent=2; return empty list if file not found.

### 2.2 Working Memory (Agent Runtime State)
- Store current task, reasoning steps, calculation results
- Persist per task_id as .json file
- Functions:
  def save_working_memory(task_id: str, data: dict) -> None
  def load_working_memory(task_id: str) -> dict
- Return empty dict if file not found.

### 2.3 Long-Term Memory (SQLite)
Database file: memory/long_term.db
Create table `user_memory` automatically:
Columns:
  id INTEGER PRIMARY KEY AUTOINCREMENT
  user_id TEXT
  key TEXT
  value TEXT
  update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP

Supported key types: risk, income, goal, taboo
Functions:
  def save_long_term(user_id: str, key: str, value: str) -> None
  def get_long_term(user_id: str, key: str) -> str | None
  def get_all_user_memory(user_id: str) -> dict

### 2.4 Knowledge Base (Chroma Local Vector DB)
Collection name: finance_rules
Functions:
  def add_knowledge(doc_id: str | int, content: str) -> None
  def query_knowledge(question: str, n_results: int = 2) -> list[str]

## 3. Technical Specifications
1. Dependencies: Only use standard Python libs + chromadb
2. Exception Handling: Add try/except for file IO, database and vector DB operations
3. Code Style: PEP8 compliant, add line comments for core logic
4. Compatibility: Pure local operation, no network request, no third-party cloud service

## 4. Agent Runtime Workflow (Logic Reference)
1. Load short-term chat history by session_id
2. Fetch user profile from long-term memory by user_id
3. Retrieve related finance knowledge via semantic search
4. Combine all memory content to build LLM prompt
5. Generate answer, then append new message to short-term memory

## 5. Output Requirement
Output the complete code of finabot/memory.py only, no extra explanation.