from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# 14 Canonical Biomarkers Contract
CANONICAL_BIOMARKERS = [
    # 10 Acoustic
    "speech_rate",
    "articulation_rate",
    "pause_frequency",
    "pause_duration_mean",
    "filled_pause_rate",
    "pitch_mean",
    "pitch_range",
    "jitter",
    "shimmer",
    "HNR",
    # 4 Linguistic / NLP
    "lexical_diversity",
    "semantic_coherence",
    "idea_density",
    "syntactic_complexity",
]

CANONICAL_BIOMARKER_SET = set(CANONICAL_BIOMARKERS)

CANONICAL_BIOMARKER_ALIASES = {
    "hnr": "HNR",
    "pause_duration": "pause_duration_mean",
    "pause_duration_sec": "pause_duration_mean",
    "speech_rate_wpm": "speech_rate",
    "articulation_rate_wpm": "articulation_rate",
}

CANONICAL_BIOMARKER_UNITS = {
    "speech_rate": "words/min",
    "articulation_rate": "words/min",
    "pause_frequency": "pauses/min",
    "pause_duration_mean": "seconds",
    "filled_pause_rate": "filled pauses/min",
    "pitch_mean": "Hz",
    "pitch_range": "Hz",
    "jitter": "%",
    "shimmer": "%",
    "HNR": "dB",
    "lexical_diversity": "MTLD score",
    "semantic_coherence": "cosine similarity",
    "idea_density": "proposition ratio",
    "syntactic_complexity": "mean tree depth",
}


def normalize_biomarker_name(name: str) -> Optional[str]:
    """Normalize a biomarker key to its canonical representation."""
    if not name:
        return None
    raw = str(name).strip()
    # Check direct canonical match
    if raw in CANONICAL_BIOMARKER_SET:
        return raw
    # Check lowercase alias
    lower = raw.lower()
    if lower in CANONICAL_BIOMARKER_ALIASES:
        return CANONICAL_BIOMARKER_ALIASES[lower]
    for canon in CANONICAL_BIOMARKERS:
        if canon.lower() == lower:
            return canon
    return None


def extract_ground_truth_value(entry: Any) -> Optional[float]:
    """Extract numeric value from float, int, dict, or TypedFeatureResult."""
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        val = entry.get("value")
        if val is not None and isinstance(val, (int, float)):
            return float(val)
        return None
    if hasattr(entry, "value"):
        val = getattr(entry, "value")
        if val is not None and isinstance(val, (int, float)):
            return float(val)
        return None
    return None


class BiomarkerExplanationItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    biomarker_id: str = Field(..., description="Canonical biomarker identifier")
    observed_value: Optional[float] = Field(None, description="Exact verified ground truth value")
    unit: Optional[str] = Field(None, description="Standard engineering unit")
    status: Optional[str] = Field(None, description="Clinical status flag, e.g., elevated, reduced, normal, anomalous")
    explanation: str = Field(..., description="Clinical contextual explanation grounded in evidence")
    clinical_context: Optional[str] = Field(None, description="Cognitive-linguistic or neuro-acoustic rationale")

    @field_validator("biomarker_id")
    @classmethod
    def validate_canonical_id(cls, v: str) -> str:
        norm = normalize_biomarker_name(v)
        if not norm:
            raise ValueError(f"Biomarker '{v}' is not one of the 14 canonical CogniSafe biomarkers.")
        return norm


class AnomalyExplanationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: Optional[int] = Field(None, description="Analyzed session ID if available")
    risk_tier: Optional[str] = Field(None, description="Verified session risk tier (Green, Yellow, Orange, Red, or null)")
    explanations: List[BiomarkerExplanationItem] = Field(default_factory=list, description="Ground-truth validated biomarker explanations")
    summary: str = Field(..., description="Synthesis summary of observed patterns")
    recommendations: List[str] = Field(default_factory=list, description="Actionable non-diagnostic recommendations")
    is_fallback: bool = Field(False, description="True if response generated via deterministic template fallback")

    def filter_and_bind_ground_truth(
        self,
        ground_truth: Optional[Dict[str, Any]] = None,
        enforce_risk_tier: Optional[str] = None,
    ) -> "AnomalyExplanationResponse":
        """
        Anti-Hallucination Guardrail:
        1. Discards any biomarker not in canonical 14.
        2. If ground_truth provided, discards any biomarker not present in ground_truth.
        3. Forces observed_value to match ground truth measurement (overwriting hallucinations).
        4. Enforces canonical units.
        5. Protects risk tier against LLM fabrication.
        """
        if enforce_risk_tier is not None:
            self.risk_tier = enforce_risk_tier

        if not ground_truth:
            return self

        normalized_gt: Dict[str, Optional[float]] = {}
        for k, v in ground_truth.items():
            norm_k = normalize_biomarker_name(k)
            if norm_k:
                normalized_gt[norm_k] = extract_ground_truth_value(v)

        validated_items: List[BiomarkerExplanationItem] = []
        for item in self.explanations:
            norm_id = normalize_biomarker_name(item.biomarker_id)
            if not norm_id or norm_id not in normalized_gt:
                # Discard unrecognized or unmeasured hallucinated biomarker
                continue

            gt_val = normalized_gt[norm_id]
            # Bind to ground-truth numerical measurement and canonical unit
            item.biomarker_id = norm_id
            item.observed_value = gt_val
            item.unit = CANONICAL_BIOMARKER_UNITS.get(norm_id, item.unit)
            validated_items.append(item)

        self.explanations = validated_items
        return self


class ExplainRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: Optional[int] = Field(None, description="Session ID to fetch ground truth from database")
    biomarkers: Optional[Dict[str, Any]] = Field(None, description="Explicit dictionary of biomarker measurements")
    anomalies: Optional[List[str]] = Field(None, description="List of biomarker names flagged as anomalous")
    baseline: Optional[Dict[str, Any]] = Field(None, description="User baseline statistics")
    risk_tier: Optional[str] = Field(None, description="Verified risk tier")


class WeeklySummaryRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_ids: Optional[List[int]] = Field(None, description="Optional list of session IDs to aggregate")
    sessions_data: Optional[List[Dict[str, Any]]] = Field(None, description="Explicit list of session biomarker maps")
    start_date: Optional[str] = Field(None, description="ISO start date string")
    end_date: Optional[str] = Field(None, description="ISO end date string")


class WeeklySummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    overall_narrative: str = Field(..., description="Longitudinal narrative synthesis for the week")
    key_observations: List[str] = Field(default_factory=list, description="Bulleted observations of acoustic & linguistic shifts")
    trend_analysis: str = Field(..., description="Directional trajectory relative to baseline")
    caregiver_notes: Optional[str] = Field(None, description="Accessible notes tailored for caregivers")
    recommendations: List[str] = Field(default_factory=list, description="Lifestyle & monitoring recommendations")
    is_fallback: bool = Field(False, description="True if response generated via deterministic template fallback")


class CaregiverSummaryRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: Optional[int] = Field(None, description="Session ID to translate")
    biomarkers: Optional[Dict[str, Any]] = Field(None, description="Explicit biomarker measurements")
    risk_tier: Optional[str] = Field(None, description="Verified risk tier")
    share_token: Optional[str] = Field(None, description="Caregiver share token for token-authorized access")
    patient_name: Optional[str] = Field("your loved one", description="Name or relation to personalize narrative")


class CaregiverSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    plain_language_summary: str = Field(..., description="Warm, jargon-free summary for family members")
    observed_changes: List[str] = Field(default_factory=list, description="Gentle summary of voice and pacing characteristics")
    supportive_tips: List[str] = Field(default_factory=list, description="Positive, practical communication guidance")
    is_fallback: bool = Field(False, description="True if generated via deterministic template fallback")


class ChatMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    messages: List[ChatMessage] = Field(..., description="List of previous conversation turns")
    system_prompt: Optional[str] = Field(None, description="Optional caller system prompt (guardrails will be prepended)")
    context_biomarkers: Optional[Dict[str, Any]] = Field(None, description="Optional session biomarker context")


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    reply: str = Field(..., description="Assistant response text")
    is_fallback: bool = Field(False, description="True if generated via fallback mechanism")
    model: str = Field("nvidia/Nemotron-3_5-Lightning", description="Model identifier used for inference")
