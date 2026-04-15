"""Hallucination Check — lightweight fact-checking and confidence scoring.

Analyzes AI agent responses for common hallucination signals: vague hedging,
unsupported claims, fabricated citations, contradictory statements, and
overconfident assertions without evidence. Returns a confidence score and
flags suspicious content.

Deploy via Chekk:
    POST https://chekk.dev/api/v1/deploy
    {"github_url": "https://github.com/Timi0217/hallucination-check"}
"""

import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Hallucination Check",
    description="Detect hallucination signals in AI agent responses",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Hallucination Signal Patterns ─────────────────────────────────────
HALLUCINATION_SIGNALS = {
    "fabricated_citation": {
        "patterns": [
            re.compile(r"(?i)according\s+to\s+(?:a\s+)?(?:\d{4}\s+)?(?:study|research|report|survey)"),
            re.compile(r"(?i)(?:published|reported)\s+in\s+(?:the\s+)?(?:journal|proceedings)"),
            re.compile(r"(?i)(?:Dr\.|Professor|Prof\.)\s+[A-Z][a-z]+\s+(?:et\s+al|and\s+colleagues)"),
            re.compile(r"(?i)(?:ISBN|DOI|PMID|arXiv)\s*[:# ]?\s*[\d.-]+"),
        ],
        "weight": 0.6,
        "description": "Possible fabricated citations or references",
    },
    "overconfident_assertion": {
        "patterns": [
            re.compile(r"(?i)(?:it\s+is|this\s+is)\s+(?:a\s+)?(?:well-known|established|proven|scientific)\s+fact"),
            re.compile(r"(?i)(?:experts?\s+)?(?:universally|unanimously|widely)\s+agree"),
            re.compile(r"(?i)there\s+is\s+no\s+(?:doubt|question|debate)\s+that"),
            re.compile(r"(?i)(?:100|99|98)%\s+(?:of\s+)?(?:experts?|scientists?|researchers?)"),
        ],
        "weight": 0.5,
        "description": "Overconfident claims without evidence",
    },
    "vague_attribution": {
        "patterns": [
            re.compile(r"(?i)(?:some|many|most|several)\s+(?:studies|experts?|researchers?|sources?)\s+(?:say|suggest|indicate|show)"),
            re.compile(r"(?i)(?:it\s+(?:has\s+been|is)\s+)?(?:widely|generally|commonly)\s+(?:believed|accepted|known)"),
            re.compile(r"(?i)research\s+(?:has\s+)?(?:shown|demonstrated|proven)\s+that"),
        ],
        "weight": 0.3,
        "description": "Vague attributions without specific sources",
    },
    "specific_numbers": {
        "patterns": [
            re.compile(r"\b\d{1,3}(?:\.\d+)?%\s+(?:of|increase|decrease|more|less|higher|lower)"),
            re.compile(r"(?i)approximately\s+\d[\d,.]+\s+(?:million|billion|trillion|people|users|cases)"),
            re.compile(r"(?i)(?:founded|established|created)\s+in\s+\d{4}"),
        ],
        "weight": 0.4,
        "description": "Specific numbers that may be fabricated",
    },
    "contradiction_signals": {
        "patterns": [
            re.compile(r"(?i)(?:however|but|although|despite)\s+(?:this|that|the\s+above)"),
            re.compile(r"(?i)(?:on\s+the\s+other\s+hand|conversely|in\s+contrast)"),
            re.compile(r"(?i)(?:it\s+is\s+worth\s+noting|it\s+should\s+be\s+noted)\s+that\s+(?:the\s+)?(?:opposite|contrary)"),
        ],
        "weight": 0.2,
        "description": "Potential self-contradictions",
    },
    "temporal_confusion": {
        "patterns": [
            re.compile(r"(?i)(?:as\s+of|since|until)\s+(?:20[3-9]\d|2[1-9]\d{2})"),  # future dates
            re.compile(r"(?i)(?:recently|just\s+last\s+(?:week|month|year))\s+(?:announced|released|published)"),
        ],
        "weight": 0.5,
        "description": "Temporal inconsistencies or future dates",
    },
}


# ── Models ────────────────────────────────────────────────────────────
class CheckRequest(BaseModel):
    text: str
    context: str | None = None  # original question/prompt for comparison
    threshold: float = 0.5


class CheckResponse(BaseModel):
    is_reliable: bool
    confidence_score: float  # 0-1, higher = more reliable
    hallucination_risk: float  # 0-1, higher = more likely hallucinated
    signals: list[dict]
    recommendation: str


class BatchCheckRequest(BaseModel):
    texts: list[str]
    threshold: float = 0.5


class BatchCheckResponse(BaseModel):
    results: list[dict]
    average_risk: float


# ── Routes ────────────────────────────────────────────────────────────

from fastapi.responses import PlainTextResponse
from pathlib import Path


@app.get("/llms.txt", response_class=PlainTextResponse)
@app.get("/.well-known/llms.txt", response_class=PlainTextResponse)
def llms_txt():
    return (Path(__file__).parent / "llms.txt").read_text()


@app.get("/")
def home():
    return {
        "service": "Hallucination Check",
        "version": "1.0.0",
        "endpoints": {
            "POST /check": "Analyze text for hallucination signals",
            "POST /batch": "Analyze multiple texts at once",
            "GET /signals": "List detection signal categories",
        },
    }


@app.post("/check", response_model=CheckResponse)
def check(req: CheckRequest):
    """Analyze text for hallucination signals and return confidence score."""
    signals = _detect_signals(req.text)

    # Calculate risk score
    if not signals:
        risk = 0.0
    else:
        total_weight = sum(s["weight"] for s in signals)
        risk = min(total_weight / 2.0, 1.0)  # normalize

    # If context provided, check for relevance
    if req.context and req.text:
        context_words = set(req.context.lower().split())
        response_words = set(req.text.lower().split())
        overlap = len(context_words & response_words) / max(len(context_words), 1)
        if overlap < 0.1:
            risk = min(risk + 0.3, 1.0)
            signals.append({
                "category": "low_relevance",
                "description": "Response has very low overlap with the question",
                "weight": 0.3,
                "match_count": 1,
            })

    confidence = round(1.0 - risk, 2)
    risk = round(risk, 2)
    is_reliable = risk < req.threshold

    # Generate recommendation
    if risk >= 0.7:
        rec = "HIGH RISK: Multiple hallucination signals detected. Verify all claims with external sources before trusting this response."
    elif risk >= 0.4:
        rec = "MODERATE RISK: Some hallucination signals detected. Cross-reference specific claims and citations."
    elif risk > 0:
        rec = "LOW RISK: Minor hallucination signals detected. Response is likely reliable but spot-check any specific numbers or citations."
    else:
        rec = "MINIMAL RISK: No significant hallucination signals detected."

    return CheckResponse(
        is_reliable=is_reliable,
        confidence_score=confidence,
        hallucination_risk=risk,
        signals=signals,
        recommendation=rec,
    )


@app.post("/batch", response_model=BatchCheckResponse)
def batch_check(req: BatchCheckRequest):
    """Analyze multiple texts for hallucination signals."""
    results = []
    total_risk = 0.0

    for text in req.texts:
        signals = _detect_signals(text)
        if not signals:
            risk = 0.0
        else:
            total_weight = sum(s["weight"] for s in signals)
            risk = min(total_weight / 2.0, 1.0)

        total_risk += risk
        results.append({
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "hallucination_risk": round(risk, 2),
            "is_reliable": risk < req.threshold,
            "signal_count": len(signals),
        })

    avg_risk = round(total_risk / max(len(req.texts), 1), 2)
    return BatchCheckResponse(results=results, average_risk=avg_risk)


@app.get("/signals")
def signals():
    """List all hallucination signal categories."""
    return {
        "signals": {
            cat_id: {
                "description": info["description"],
                "weight": info["weight"],
                "pattern_count": len(info["patterns"]),
            }
            for cat_id, info in HALLUCINATION_SIGNALS.items()
        }
    }


def _detect_signals(text: str) -> list[dict]:
    """Scan text for all hallucination signal patterns."""
    signals = []
    for cat_id, info in HALLUCINATION_SIGNALS.items():
        match_count = 0
        for pattern in info["patterns"]:
            matches = pattern.findall(text)
            match_count += len(matches)
        if match_count > 0:
            signals.append({
                "category": cat_id,
                "description": info["description"],
                "weight": info["weight"],
                "match_count": match_count,
            })
    return signals
