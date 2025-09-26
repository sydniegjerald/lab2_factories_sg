from typing import List, Dict, Any

def make_vector(subject: str, body: str) -> List[float]:
    subj_len = len(subject or "")
    body_len = len(body or "")
    return [float(subj_len), float(body_len)]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = (a[0] * b[0]) + (a[1] * b[1])
    norm_a = (a[0]**2 + a[1]**2) ** 0.5
    norm_b = (b[0]**2 + b[1]**2) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

def find_nearest_email(
    subject: str,
    body: str,
    stored_emails: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if not stored_emails:
        return {"predicted_class": None, "match": None, "similarity": 0.0}

    query_vec = make_vector(subject, body)
    best_email = None
    best_score = -1.0

    for e in stored_emails:
        vec = make_vector(e.get("subject", ""), e.get("body", ""))
        score = cosine_similarity(query_vec, vec)
        if score > best_score:
            best_score = score
            best_email = e

    return {
        "predicted_class": best_email.get("ground_truth") if best_email else None,
        "match": best_email,
        "similarity": best_score
    }
