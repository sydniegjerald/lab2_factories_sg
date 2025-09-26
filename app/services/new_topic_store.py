import json
from pathlib import Path
from typing import Dict, Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TOPICS_PATH = DATA_DIR / "topic_keywords.json"

def load_topics() -> Dict[str, Dict[str, Any]]:
    text = TOPICS_PATH.read_text(encoding="utf-8")
    return json.loads(text)

def save_topics(data: Dict[str, Dict[str, Any]]) -> None:
    #write topics dict back to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOPICS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

def add_topic(name: str, description: str) -> Dict[str, Dict[str, Any]]:
    topics = load_topics()
    if name in topics:
        raise ValueError(f"Topic '{name}' already exists.")
    topics[name] = {"description": description}
    save_topics(topics)
    return topics
