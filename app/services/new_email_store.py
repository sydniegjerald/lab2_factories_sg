import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EMAILS_PATH = DATA_DIR / "emails.json"

@dataclass
class StoredEmail:
    subject: str
    body: str
    ground_truth: Optional[str] = None

def _ensure_file_exists() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EMAILS_PATH.exists():
        EMAILS_PATH.write_text("[]", encoding="utf-8")

def list_emails() -> List[Dict[str, Any]]:
    _ensure_file_exists() #
    text = EMAILS_PATH.read_text(encoding="utf-8")
    return json.loads(text)

def add_email(subject: str, body: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:

    _ensure_file_exists()
    items = list_emails() ##
    record = asdict(StoredEmail(subject=subject, body=body, ground_truth=ground_truth))
    items.append(record)
    EMAILS_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return record
