from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
from typing import Optional
from app.services.new_email_store import add_email, list_emails
from app.services.new_topic_store import load_topics, add_topic as add_topic_to_store
from app.models.new_similarity_method import find_nearest_email


router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email)
        
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics for my ML Ops Class"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()



# -- HW assignment additions

class TopicCreateRequest(BaseModel):
    name: str
    description: str

class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str] = None  # optional label like "promotion", "work", etc.

class SimpleClassifyRequest(BaseModel):
    subject: str
    body: str
    mode: str = "topic"  # "topic" (existing pipeline) or "nearest_email" (copy label from similar stored email)

@router.post("/topics_hw")
async def create_topic(payload: TopicCreateRequest):
    """Create a new topic -- for hw assignment"""
    try:
        topics = add_topic_to_store(payload.name, payload.description)
        return topics
    except ValueError as ve:
        # raised if topic already exists
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/emails_hw")
async def get_emails():
    """List stored emails -- for hw assignment"""
    try:
        return list_emails()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emails_hw")
async def post_email(payload: EmailStoreRequest):
    """Store a new email -- for hw assignment"""
    try:
        # If a label is provided, ensure it's a known topic
        if payload.ground_truth:
            topics = load_topics()  # {"work": {"description": ...}, ...}
            if payload.ground_truth not in topics:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown topic '{payload.ground_truth}'. "
                           f"Create it first or choose one of: {list(topics.keys())}"
                )

        saved = add_email(
            subject=payload.subject,
            body=payload.body,
            ground_truth=payload.ground_truth
        )
        return saved
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify_hw")
async def classify_simple(request: SimpleClassifyRequest):
    """Classify an email by topic or nearest stored email-- for hw assignment"""
    try:
        if request.mode not in ("topic", "nearest_email"):
            raise HTTPException(status_code=400, detail="mode must be 'topic' or 'nearest_email'")

        if request.mode == "topic":
            # Use your existing topic pipeline (unchanged)
            service = EmailTopicInferenceService()
            email = Email(subject=request.subject, body=request.body)
            result = service.classify_email(email)
            return {
                "mode": "topic",
                "predicted_class": result.get("predicted_topic"),
                "topic_scores": result.get("topic_scores"),
                "features": result.get("features"),
                "available_topics": result.get("available_topics"),
                "note": "Predicted by comparing to topic descriptions."
            }

        # nearest_email mode: copy label from the most similar stored email
        stored = list_emails()
        best = find_nearest_email(request.subject, request.body, stored)

        if best["predicted_class"] is None:
            return {
                "mode": "nearest_email",
                "predicted_class": None,
                "reason": "No stored emails or closest match has no label.",
                "similarity": best["similarity"],
                "matched_email": best["match"]
            }

        return {
            "mode": "nearest_email",
            "predicted_class": best["predicted_class"],
            "similarity": best["similarity"],
            "matched_email": best["match"],
            "note": "Predicted by copying the label of the most similar stored email."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
