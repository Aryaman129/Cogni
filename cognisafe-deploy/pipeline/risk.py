"""
risk.py — Biomarker merging and rule-based clinical risk scoring.

Implements Section 7 requirements:
1. Merge acoustic and NLP biomarkers preserving None for missing/invalid measurements.
   Eliminates silent 0.0 fallbacks.
2. Decouple analysis_status from risk_tier:
   - If user has < 5 sessions: analysis_status = "insufficient_baseline", risk_tier = None.
   - If input features are incomplete or invalid: analysis_status = "insufficient_data", risk_tier = None.
   - Never default to "Green" when risk tier is null or missing.
3. Rule-based clinical risk scoring based on statistical anomaly deviations (>=2.5σ or 3.0σ).
   Decommissions synthetic XGBoost from clinical risk determination.
4. Full integration with TypedFeatureResult and BiomarkerCollectionResult.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from pipeline.schemas import (
        CANONICAL_ACOUSTIC_BIOMARKERS,
        CANONICAL_NLP_BIOMARKERS,
        CANONICAL_BIOMARKERS,
        FeatureStatus,
        TypedFeatureResult,
        BiomarkerCollectionResult,
    )
    from pipeline.baseline import extract_numeric_value, is_baseline_mature
except ImportError:
    from schemas import (
        CANONICAL_ACOUSTIC_BIOMARKERS,
        CANONICAL_NLP_BIOMARKERS,
        CANONICAL_BIOMARKERS,
        FeatureStatus,
        TypedFeatureResult,
        BiomarkerCollectionResult,
    )
    from baseline import extract_numeric_value, is_baseline_mature

CANONICAL_RISK_TIERS: List[str] = ["Green", "Yellow", "Orange", "Red"]
TIER_ORDER: Dict[str, int] = {"Green": 0, "Yellow": 1, "Orange": 2, "Red": 3}


def merge_biomarkers(
    acoustic: Union[Dict[str, Any], BiomarkerCollectionResult, None],
    nlp: Union[Dict[str, Any], BiomarkerCollectionResult, None],
    return_typed: bool = False,
) -> Union[Dict[str, Optional[float]], BiomarkerCollectionResult]:
    """
    Merge acoustic and NLP biomarkers into the canonical 14-biomarker format.

    Guarantees:
    - Missing or failed features are represented as None (not silent 0.0).
    - If return_typed=True, returns BiomarkerCollectionResult containing TypedFeatureResult.
    - If return_typed=False, returns standard dict {biomarker: float_or_None}.
    """
    acoustic_map: Dict[str, Any] = {}
    if isinstance(acoustic, BiomarkerCollectionResult):
        acoustic_map = acoustic.features
    elif isinstance(acoustic, dict):
        acoustic_map = acoustic

    nlp_map: Dict[str, Any] = {}
    if isinstance(nlp, BiomarkerCollectionResult):
        nlp_map = nlp.features
    elif isinstance(nlp, dict):
        nlp_map = nlp

    if return_typed:
        collection = BiomarkerCollectionResult()
        # Merge acoustic features
        for bm in CANONICAL_ACOUSTIC_BIOMARKERS:
            val = acoustic_map.get(bm)
            if isinstance(val, TypedFeatureResult):
                collection[bm] = val
            elif val is not None:
                num = extract_numeric_value(val)
                if num is not None:
                    collection[bm] = TypedFeatureResult(
                        status=FeatureStatus.VALID,
                        value=num,
                        unit="unknown",
                        method="merged_acoustic",
                    )
                else:
                    collection[bm] = TypedFeatureResult(
                        status=FeatureStatus.INSUFFICIENT_INPUT,
                        value=None,
                        unit="unknown",
                        method="merged_acoustic",
                        reason="non_numeric_or_invalid",
                    )
            else:
                collection[bm] = TypedFeatureResult(
                    status=FeatureStatus.INSUFFICIENT_INPUT,
                    value=None,
                    unit="unknown",
                    method="merged_acoustic",
                    reason="missing_feature",
                )

        # Merge NLP features
        for bm in CANONICAL_NLP_BIOMARKERS:
            val = nlp_map.get(bm)
            if isinstance(val, TypedFeatureResult):
                collection[bm] = val
            elif val is not None:
                num = extract_numeric_value(val)
                if num is not None:
                    collection[bm] = TypedFeatureResult(
                        status=FeatureStatus.VALID,
                        value=num,
                        unit="unknown",
                        method="merged_nlp",
                    )
                else:
                    collection[bm] = TypedFeatureResult(
                        status=FeatureStatus.INSUFFICIENT_INPUT,
                        value=None,
                        unit="unknown",
                        method="merged_nlp",
                        reason="non_numeric_or_invalid",
                    )
            else:
                collection[bm] = TypedFeatureResult(
                    status=FeatureStatus.INSUFFICIENT_INPUT,
                    value=None,
                    unit="unknown",
                    method="merged_nlp",
                    reason="missing_feature",
                )

        # Set overall status
        if collection.valid_count == len(CANONICAL_BIOMARKERS):
            collection.overall_status = "complete"
        elif collection.valid_count == 0:
            collection.overall_status = "failed"
        else:
            collection.overall_status = "partial"
        return collection

    # Return standard dictionary mapping biomarker -> Optional[float]
    result: Dict[str, Optional[float]] = {}
    for bm in CANONICAL_ACOUSTIC_BIOMARKERS:
        val = acoustic_map.get(bm)
        result[bm] = extract_numeric_value(val)

    for bm in CANONICAL_NLP_BIOMARKERS:
        val = nlp_map.get(bm)
        result[bm] = extract_numeric_value(val)

    return result


def calculate_clinical_risk(
    anomaly_count: int,
    baseline_mature: bool,
    severe_count: Optional[int] = None,
) -> Optional[str]:
    """
    Calculate clinical risk tier strictly based on personal baseline statistical anomalies.

    Contract:
    - If baseline_mature is False: ALWAYS returns None.
    - If severe_count is supplied:
        severe >= 3: "Red"
        severe == 2: "Orange"
        severe == 1: "Yellow"
        severe == 0: "Green" (if anomaly_count == 0) else "Yellow" / "Orange"
    - If severe_count is None (matching standard E2E test tier contract):
        0 anomalies: "Green"
        1-2 anomalies: "Yellow"
        3-4 anomalies: "Orange"
        5+ anomalies: "Red"
    """
    if not baseline_mature:
        return None

    if severe_count is not None:
        if severe_count >= 3:
            return "Red"
        elif severe_count == 2:
            return "Orange"
        elif severe_count == 1:
            return "Yellow"
        else:
            # severe_count == 0
            if anomaly_count == 0:
                return "Green"
            elif anomaly_count <= 2:
                return "Yellow"
            elif anomaly_count <= 4:
                return "Orange"
            return "Red"

    # Default rule-based anomaly count evaluation
    if anomaly_count == 0:
        return "Green"
    elif anomaly_count <= 2:
        return "Yellow"
    elif anomaly_count <= 4:
        return "Orange"
    return "Red"


def evaluate_risk_tier(
    anomaly_flags: List[Dict[str, Any]],
    baseline_mature: bool,
) -> Optional[str]:
    """
    Evaluate clinical risk tier from a list of anomaly flag dictionaries.
    Each flag has: {biomarker, severity ('mild'|'moderate'|'severe'), deviation, ...}

    Rules:
    - If not baseline_mature: returns None (never default to Green).
    - If empty flags: returns "Green".
    - Counting severe deviations (>=2.5σ or >=3.0σ):
        Green: 0 anomalies
        Yellow: 1 severe/moderate anomaly or mild anomalies
        Orange: 2 severe/moderate anomalies
        Red: >=3 severe/moderate anomalies
    """
    if not baseline_mature:
        return None

    if not anomaly_flags:
        return "Green"

    severe_count = sum(
        1 for f in anomaly_flags
        if f.get("severity") == "severe" or f.get("deviation", 0) >= 3.0
    )
    moderate_count = sum(
        1 for f in anomaly_flags
        if f.get("severity") == "moderate" or 2.5 <= f.get("deviation", 0) < 3.0
    )
    mild_count = sum(
        1 for f in anomaly_flags
        if f.get("severity") == "mild" or 2.0 <= f.get("deviation", 0) < 2.5
    )

    combined_severe = severe_count + moderate_count

    if combined_severe >= 3:
        return "Red"
    elif combined_severe == 2:
        return "Orange"
    elif combined_severe == 1 or mild_count >= 1:
        return "Yellow"
    return "Green"


def determine_analysis_status(
    features: Union[Dict[str, Any], BiomarkerCollectionResult],
    session_count: int,
    min_sessions_for_maturity: int = 5,
    mode: str = "voice",
    has_extraction_errors: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Decouple analysis_status from risk_tier.

    Returns:
        (analysis_status, risk_tier)

    States:
        - "insufficient_data": Audio empty/silent or missing required mode features. risk_tier = None.
        - "analysis_failed": Critical extraction failure. risk_tier = None.
        - "insufficient_baseline": Valid features extracted, but user has < 5 sessions. risk_tier = None.
        - "complete": Valid features and >= 5 sessions. risk_tier calculated downstream.
    """
    if has_extraction_errors:
        return ("analysis_failed", None)

    # Check valid feature counts
    if isinstance(features, BiomarkerCollectionResult):
        valid_count = features.valid_count
        valid_acoustic = sum(
            1 for bm in CANONICAL_ACOUSTIC_BIOMARKERS
            if bm in features and features[bm].status == FeatureStatus.VALID and features[bm].value is not None
        )
    elif isinstance(features, dict):
        valid_count = sum(1 for v in features.values() if extract_numeric_value(v) is not None)
        valid_acoustic = sum(
            1 for bm in CANONICAL_ACOUSTIC_BIOMARKERS
            if bm in features and extract_numeric_value(features[bm]) is not None
        )
    else:
        return ("insufficient_data", None)

    # If in voice mode and no acoustic features could be measured (e.g. silent audio)
    if mode == "voice" and valid_acoustic == 0:
        return ("insufficient_data", None)

    # If completely empty or no valid features
    if valid_count == 0:
        return ("insufficient_data", None)

    # Check baseline maturity threshold
    if not is_baseline_mature(session_count, threshold=min_sessions_for_maturity):
        return ("insufficient_baseline", None)

    return ("complete", None)