import os
import json
import logging
from typing import Optional, Dict, Any, List, Union
import httpx

from schemas.ai import (
    CANONICAL_BIOMARKERS,
    CANONICAL_BIOMARKER_UNITS,
    normalize_biomarker_name,
    extract_ground_truth_value,
    BiomarkerExplanationItem,
    AnomalyExplanationResponse,
    WeeklySummaryResponse,
    CaregiverSummaryResponse,
    ChatResponse,
)

logger = logging.getLogger(__name__)

SYSTEM_GUARDRAILS = (
    "You are an expert clinical neuroscience and speech-language biomarker AI assistant embedded in CogniSafe.\n"
    "CRITICAL CLINICAL & ANTI-HALLUCINATION RULES:\n"
    "1. You must NEVER calculate, invent, extrapolate, or alter any biomarker numerical values, units, baselines, or risk tiers.\n"
    "2. You may ONLY reference the exact ground-truth biomarker measurements, units, and risk tiers explicitly provided in the prompt.\n"
    "3. If a biomarker is missing, unmeasured, or invalid, state that it was not measured; never fabricate an estimate.\n"
    "4. CogniSafe is a voice-first cognitive-health monitoring platform, NOT a diagnostic medical device. Never state or imply a definitive medical diagnosis (such as Alzheimer's disease or dementia). Always recommend consulting a qualified neurologist or physician for clinical evaluation.\n"
    "5. Output must be strictly valid JSON according to the schema requested."
)

CLINICAL_DESCRIPTIONS = {
    "speech_rate": "Overall rate of speech in words per minute. Slowing of speech rate can reflect increased cognitive planning latency, lexical retrieval effort, or motor speech slowing.",
    "articulation_rate": "Pacing of speech excluding pause intervals. Reductions highlight lengthened phonetic duration independent of hesitation pauses.",
    "pause_frequency": "Frequency of conversational pause intervals per minute. Elevated pause frequency often marks hesitation during word-finding or sentence planning.",
    "pause_duration_mean": "Mean duration of pause intervals. Prolonged hesitations suggest difficulties in conceptual formulation or lexical access.",
    "filled_pause_rate": "Rate of filled pauses (e.g., 'um', 'uh'). Frequent fillers reflect active lexical search and speech formulation monitoring.",
    "pitch_mean": "Mean fundamental frequency (F0) during voiced speech, representing baseline vocal pitch.",
    "pitch_range": "Variability in fundamental frequency across voiced frames. A restricted pitch range reflects acoustic monopitch or diminished prosodic modulation.",
    "jitter": "Micro-perturbations in vocal fold vibration frequency. Elevated jitter indicates acoustic perturbation in vocal fold mechanics.",
    "shimmer": "Cycle-to-cycle amplitude perturbations across voiced phonatory frames. Elevated shimmer indicates phonatory amplitude instability.",
    "HNR": "Harmonics-to-noise ratio in decibels. Lower HNR reflects increased relative spectral noise or acoustic breathiness in vocal quality.",
    "lexical_diversity": "Measure of vocabulary richness (MTLD). Reductions indicate lexical repetition and vocabulary shrinkage.",
    "semantic_coherence": "Sentence-to-sentence embedding cosine similarity. Lower coherence indicates thematic drift or reduced semantic progression.",
    "idea_density": "Propositional density indicating information content per utterance. Reductions reflect simplified conceptual communication.",
    "syntactic_complexity": "Mean dependency parse tree depth. Reductions reflect simplified grammatical and structural sentence complexity.",
}


class NebiusClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("NEBIUS_API_KEY", "")
        self.base_url = (
            base_url
            if base_url is not None
            else os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
        ).rstrip("/")
        self.model_id = (
            model_id
            if model_id is not None
            else os.getenv("NEBIUS_MODEL_ID", "nvidia/Nemotron-3_5-Lightning")
        )
        try:
            self.timeout_seconds = (
                float(timeout_seconds)
                if timeout_seconds is not None
                else float(os.getenv("NEBIUS_TIMEOUT_SECONDS", "30.0"))
            )
        except (ValueError, TypeError):
            self.timeout_seconds = 30.0

        self._custom_http_client = http_client

    def is_configured(self) -> bool:
        """Return True if a non-empty API key is present."""
        return bool(self.api_key and self.api_key.strip())

    async def _call_chat_completions(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Low-level OpenAI-compatible completions call to Nebius Token Factory."""
        if not self.is_configured():
            raise ValueError("Nebius API key is not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        if self._custom_http_client:
            res = await self._custom_http_client.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            res.raise_for_status()
            return res.json()

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            res = await client.post(url, headers=headers, json=payload)
            res.raise_for_status()
            return res.json()

    @staticmethod
    def _extract_json_content(content_str: Optional[str]) -> Dict[str, Any]:
        """Safely extract JSON object from LLM response text, stripping markdown codeblocks and thinking preambles."""
        import re
        if not content_str:
            raise ValueError("Empty or None content returned from completions call.")
        text = content_str.strip()

        # Strategy 1: Find all markdown codeblocks containing JSON, inspecting from latest to earliest
        codeblocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        for cb in reversed(codeblocks):
            cb_clean = cb.strip()
            if cb_clean.startswith("{") and cb_clean.endswith("}"):
                try:
                    data = json.loads(cb_clean)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass

        # Strategy 2: Find outermost balanced { ... } candidates
        candidates = []
        first_idx = text.find("{")
        while first_idx != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(first_idx, len(text)):
                c = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[first_idx:i+1]
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict) and len(parsed) > 0:
                                    candidates.append(parsed)
                            except Exception:
                                pass
                            break
            first_idx = text.find("{", first_idx + 1)

        if candidates:
            # Prefer candidates that have target schema keys
            for c in reversed(candidates):
                if any(k in c for k in ("explanations", "summary", "plain_language_summary", "overall_narrative", "recommendations")):
                    return c
            return candidates[-1]

        # Fallback to direct json.loads
        return json.loads(text)

    # --------------------------------------------------------------------------
    # 1. Anomaly & Biomarker Explanations
    # --------------------------------------------------------------------------
    async def explain_biomarkers(
        self,
        biomarkers: Dict[str, Any],
        anomalies: Optional[List[str]] = None,
        baseline: Optional[Dict[str, Any]] = None,
        risk_tier: Optional[str] = None,
        session_id: Optional[int] = None,
    ) -> AnomalyExplanationResponse:
        """
        Generate contextual clinical explanations for anomalous biomarkers with strict
        ground-truth binding and seamless deterministic fallback.
        """
        # Ground truth normalization
        ground_truth: Dict[str, Optional[float]] = {}
        for k, v in (biomarkers or {}).items():
            norm_k = normalize_biomarker_name(k)
            if norm_k:
                ground_truth[norm_k] = extract_ground_truth_value(v)

        if not self.is_configured():
            logger.info("Nebius API key unconfigured; using deterministic template explanation.")
            return self._deterministic_anomaly_explanation(
                ground_truth, anomalies, baseline, risk_tier, session_id
            )

        try:
            prompt_data = {
                "verified_ground_truth_biomarkers": {
                    k: {"value": v, "unit": CANONICAL_BIOMARKER_UNITS.get(k, "")}
                    for k, v in ground_truth.items()
                },
                "flagged_anomalies": anomalies or [],
                "session_risk_tier": risk_tier or "Unassigned",
                "baseline_statistics": baseline or {},
            }

            user_prompt = (
                f"Analyze the following verified ground-truth speech biomarkers and flagged anomalies:\n"
                f"{json.dumps(prompt_data, indent=2)}\n\n"
                "Provide a structured JSON object with the following schema:\n"
                "{\n"
                '  "explanations": [\n'
                '    {"biomarker_id": "...", "observed_value": 0.0, "unit": "...", "status": "elevated/reduced/normal", "explanation": "...", "clinical_context": "..."}\n'
                "  ],\n"
                '  "summary": "Overall synthesis paragraph",\n'
                '  "recommendations": ["Recommendation 1", "Recommendation 2"]\n'
                "}\n"
                "STRICT REQUIREMENT: Only reference the provided ground-truth biomarkers. Do not invent any numbers."
            )

            messages = [
                {"role": "system", "content": SYSTEM_GUARDRAILS},
                {"role": "user", "content": user_prompt},
            ]

            res_json = await self._call_chat_completions(
                messages,
                temperature=0.2,
                max_tokens=3000,
            )
            raw_msg = res_json["choices"][0]["message"]
            raw_reply = raw_msg.get("content") or raw_msg.get("reasoning")
            parsed = self._extract_json_content(raw_reply)

            response_obj = AnomalyExplanationResponse(
                session_id=session_id,
                risk_tier=risk_tier,
                explanations=parsed.get("explanations", []),
                summary=parsed.get("summary", "Analysis completed."),
                recommendations=parsed.get("recommendations", []),
                is_fallback=False,
            )

            # Apply Anti-Hallucination Guardrail: discard hallucinated keys and bind ground truth
            response_obj.filter_and_bind_ground_truth(ground_truth=ground_truth, enforce_risk_tier=risk_tier)
            return response_obj

        except Exception as exc:
            logger.warning("Nebius inference failed (%s); falling back to deterministic explanation.", exc)
            return self._deterministic_anomaly_explanation(
                ground_truth, anomalies, baseline, risk_tier, session_id
            )

    def _deterministic_anomaly_explanation(
        self,
        ground_truth: Dict[str, Optional[float]],
        anomalies: Optional[List[str]],
        baseline: Optional[Dict[str, Any]],
        risk_tier: Optional[str],
        session_id: Optional[int],
    ) -> AnomalyExplanationResponse:
        """Deterministic fallback explanation based on established clinical speech-language science."""
        items: List[BiomarkerExplanationItem] = []
        target_keys = set()

        if anomalies:
            for a in anomalies:
                norm = normalize_biomarker_name(a)
                if norm and norm in ground_truth:
                    target_keys.add(norm)

        # If no specific anomalies flagged, explain all measured biomarkers
        if not target_keys:
            target_keys = {k for k, v in ground_truth.items() if v is not None}

        for canon_id in CANONICAL_BIOMARKERS:
            if canon_id in target_keys:
                val = ground_truth.get(canon_id)
                unit = CANONICAL_BIOMARKER_UNITS.get(canon_id, "")
                desc = CLINICAL_DESCRIPTIONS.get(canon_id, "Speech biomarker indicator.")

                if val is not None:
                    exp_text = f"Observed {canon_id.replace('_', ' ')} of {val:.2f} {unit}. {desc}"
                    status = "anomalous" if (anomalies and canon_id in (normalize_biomarker_name(x) for x in anomalies)) else "measured"
                else:
                    exp_text = f"{canon_id.replace('_', ' ')} was not measured in this session."
                    status = "unmeasured"

                items.append(
                    BiomarkerExplanationItem(
                        biomarker_id=canon_id,
                        observed_value=val,
                        unit=unit,
                        status=status,
                        explanation=exp_text,
                        clinical_context=desc,
                    )
                )

        count = len(items)
        tier_str = f"Risk tier: {risk_tier}." if risk_tier else "Risk tier: Establishing Baseline."
        summary = (
            f"Voice biomarker assessment completed across {count} tracked acoustic and linguistic metrics. "
            f"{tier_str} Biomarker trajectories should be monitored longitudinally against personal historical baselines."
        )
        recommendations = [
            "Maintain consistent daily or weekly voice check-ins to build baseline stability.",
            "Discuss prolonged shifts in speech rate, pause patterns, or vocabulary diversity with a qualified healthcare professional.",
        ]

        return AnomalyExplanationResponse(
            session_id=session_id,
            risk_tier=risk_tier,
            explanations=items,
            summary=summary,
            recommendations=recommendations,
            is_fallback=True,
        )

    # --------------------------------------------------------------------------
    # 2. Weekly Longitudinal Summary
    # --------------------------------------------------------------------------
    async def generate_weekly_summary(
        self,
        sessions_data: List[Dict[str, Any]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> WeeklySummaryResponse:
        """
        Generate longitudinal weekly trend summary for clinicians and caregivers.
        """
        if not self.is_configured():
            logger.info("Nebius API key unconfigured; using deterministic weekly summary.")
            return self._deterministic_weekly_summary(sessions_data, start_date, end_date)

        try:
            summary_stats = self._compute_weekly_statistics(sessions_data)
            user_prompt = (
                f"Review the following weekly speech biomarker metrics ({len(sessions_data)} sessions recorded "
                f"between {start_date or 'recent days'} and {end_date or 'today'}):\n"
                f"{json.dumps(summary_stats, indent=2)}\n\n"
                "Generate a clinical longitudinal summary as JSON with:\n"
                "{\n"
                '  "overall_narrative": "...",\n'
                '  "key_observations": ["...", "..."],\n'
                '  "trend_analysis": "...",\n'
                '  "caregiver_notes": "...",\n'
                '  "recommendations": ["...", "..."]\n'
                "}\n"
                "STRICT REQUIREMENT: Ground all observations strictly in the provided summary statistics. Do not invent diagnoses."
            )

            messages = [
                {"role": "system", "content": SYSTEM_GUARDRAILS},
                {"role": "user", "content": user_prompt},
            ]

            res_json = await self._call_chat_completions(
                messages,
                temperature=0.2,
                max_tokens=3000,
            )
            raw_msg = res_json["choices"][0]["message"]
            raw_reply = raw_msg.get("content") or raw_msg.get("reasoning")
            parsed = self._extract_json_content(raw_reply)

            return WeeklySummaryResponse(
                overall_narrative=parsed.get("overall_narrative", "Weekly monitoring complete."),
                key_observations=parsed.get("key_observations", []),
                trend_analysis=parsed.get("trend_analysis", "Biomarker metrics remained within expected variance."),
                caregiver_notes=parsed.get("caregiver_notes"),
                recommendations=parsed.get("recommendations", []),
                is_fallback=False,
            )

        except Exception as exc:
            logger.warning("Nebius weekly summary failed (%s); using deterministic template.", exc)
            return self._deterministic_weekly_summary(sessions_data, start_date, end_date)

    @staticmethod
    def _compute_weekly_statistics(sessions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Helper to calculate aggregate statistics from session records."""
        total = len(sessions_data)
        if total == 0:
            return {"session_count": 0}

        risk_counts: Dict[str, int] = {}
        speech_rates: List[float] = []
        pause_freqs: List[float] = []
        coherences: List[float] = []

        for s in sessions_data:
            tier = s.get("risk_tier") or "Establishing Baseline"
            risk_counts[tier] = risk_counts.get(tier, 0) + 1

            sr = extract_ground_truth_value(s.get("speech_rate"))
            if sr is not None:
                speech_rates.append(sr)

            pf = extract_ground_truth_value(s.get("pause_frequency"))
            if pf is not None:
                pause_freqs.append(pf)

            sc = extract_ground_truth_value(s.get("semantic_coherence"))
            if sc is not None:
                coherences.append(sc)

        return {
            "session_count": total,
            "risk_tier_distribution": risk_counts,
            "speech_rate_avg": round(sum(speech_rates) / len(speech_rates), 1) if speech_rates else None,
            "pause_frequency_avg": round(sum(pause_freqs) / len(pause_freqs), 2) if pause_freqs else None,
            "semantic_coherence_avg": round(sum(coherences) / len(coherences), 3) if coherences else None,
        }

    def _deterministic_weekly_summary(
        self,
        sessions_data: List[Dict[str, Any]],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> WeeklySummaryResponse:
        stats = self._compute_weekly_statistics(sessions_data)
        count = stats.get("session_count", 0)

        if count == 0:
            return WeeklySummaryResponse(
                overall_narrative="No sessions were recorded during this period.",
                key_observations=["Zero sessions available for longitudinal trend computation."],
                trend_analysis="Insufficient sessions to establish trend trajectory.",
                caregiver_notes="Encourage completing daily voice recordings to activate longitudinal tracking.",
                recommendations=["Record voice sessions at least 3-5 times weekly."],
                is_fallback=True,
            )

        sr_avg = stats.get("speech_rate_avg")
        pf_avg = stats.get("pause_frequency_avg")
        sc_avg = stats.get("semantic_coherence_avg")
        dist = stats.get("risk_tier_distribution", {})

        obs = [f"Completed {count} voice monitoring session(s) in the evaluation window."]
        if sr_avg is not None:
            obs.append(f"Average speech rate was {sr_avg} words/min.")
        if pf_avg is not None:
            obs.append(f"Average pause frequency was {pf_avg} pauses/min.")
        if sc_avg is not None:
            obs.append(f"Mean semantic coherence was {sc_avg}.")

        trend = (
            f"Voice indicators across {count} sessions show consistent recording adherence. "
            f"Risk distribution: {', '.join(f'{k}: {v}' for k, v in dist.items())}."
        )

        return WeeklySummaryResponse(
            overall_narrative=f"Weekly longitudinal review summarizes {count} completed voice sessions. {trend}",
            key_observations=obs,
            trend_analysis=trend,
            caregiver_notes="Overall voice patterns remain steady. Continue encouraging regular, unhurried conversations.",
            recommendations=[
                "Maintain regular monitoring schedule at the same time of day.",
                "Review longitudinal trend graphs with a clinician during routine check-ups.",
            ],
            is_fallback=True,
        )

    # --------------------------------------------------------------------------
    # 3. Caregiver Plain-Language Translation
    # --------------------------------------------------------------------------
    async def generate_caregiver_summary(
        self,
        biomarkers: Optional[Dict[str, Any]] = None,
        risk_tier: Optional[str] = None,
        patient_name: str = "your loved one",
    ) -> CaregiverSummaryResponse:
        """
        Generate plain-language translation of clinical metrics for family members.
        """
        if not self.is_configured():
            logger.info("Nebius API key unconfigured; using deterministic caregiver summary.")
            return self._deterministic_caregiver_summary(biomarkers, risk_tier, patient_name)

        try:
            ground_truth = {
                k: extract_ground_truth_value(v)
                for k, v in (biomarkers or {}).items()
                if normalize_biomarker_name(k)
            }

            user_prompt = (
                f"Translate the following speech biomarker results for {patient_name} (Risk tier: {risk_tier or 'Unassigned'}):\n"
                f"{json.dumps(ground_truth, indent=2)}\n\n"
                "Create a compassionate, plain-language summary for family members without medical jargon.\n"
                "Return JSON with:\n"
                "{\n"
                '  "plain_language_summary": "...",\n'
                '  "observed_changes": ["...", "..."],\n'
                '  "supportive_tips": ["...", "..."]\n'
                "}\n"
                "CRITICAL: Do not diagnose. Keep tone supportive, encouraging, and clear."
            )

            messages = [
                {"role": "system", "content": SYSTEM_GUARDRAILS},
                {"role": "user", "content": user_prompt},
            ]

            res_json = await self._call_chat_completions(
                messages,
                temperature=0.3,
                max_tokens=2500,
            )
            raw_msg = res_json["choices"][0]["message"]
            raw_reply = raw_msg.get("content") or raw_msg.get("reasoning")
            parsed = self._extract_json_content(raw_reply)

            return CaregiverSummaryResponse(
                plain_language_summary=parsed.get("plain_language_summary", f"Voice summary for {patient_name}."),
                observed_changes=parsed.get("observed_changes", []),
                supportive_tips=parsed.get("supportive_tips", []),
                is_fallback=False,
            )

        except Exception as exc:
            logger.warning("Nebius caregiver summary failed (%s); using deterministic template.", exc)
            return self._deterministic_caregiver_summary(biomarkers, risk_tier, patient_name)

    def _deterministic_caregiver_summary(
        self,
        biomarkers: Optional[Dict[str, Any]],
        risk_tier: Optional[str],
        patient_name: str,
    ) -> CaregiverSummaryResponse:
        changes = []
        if biomarkers:
            sr = extract_ground_truth_value(biomarkers.get("speech_rate"))
            if sr is not None:
                if sr < 100:
                    changes.append(f"{patient_name}'s speech pacing was slightly slower than typical conversational speed.")
                elif sr > 150:
                    changes.append(f"{patient_name}'s speech pacing was lively and brisk.")
                else:
                    changes.append(f"{patient_name}'s speech rate was in a steady, comfortable conversational range.")

            pf = extract_ground_truth_value(biomarkers.get("pause_frequency"))
            if pf is not None and pf > 6:
                changes.append(f"{patient_name} paused more frequently during speech, which can happen when searching for words or feeling fatigued.")

        if not changes:
            changes.append(f"{patient_name}'s voice patterns were steady throughout the recent recording.")

        tips = [
            f"Offer {patient_name} ample time to finish sentences without rushing or interrupting.",
            "Choose a quiet, comfortable environment for conversations to minimize background distractions.",
            "Celebrate shared moments and focus on natural, relaxed discussions rather than testing memory.",
        ]

        summary_text = (
            f"In recent voice check-ins, {patient_name} engaged well. Voice-first monitoring tracks subtle patterns "
            f"over time to provide peace of mind and support your family's care journey."
        )

        return CaregiverSummaryResponse(
            plain_language_summary=summary_text,
            observed_changes=changes,
            supportive_tips=tips,
            is_fallback=True,
        )

    # --------------------------------------------------------------------------
    # 4. Interactive Chat
    # --------------------------------------------------------------------------
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        context_biomarkers: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """
        Interactive conversational support with system guardrails and deterministic fallback.
        """
        if not self.is_configured():
            logger.info("Nebius API key unconfigured; using fallback chat response.")
            return ChatResponse(
                reply=self._deterministic_chat_response(messages),
                is_fallback=True,
                model=self.model_id,
            )

        try:
            full_system = SYSTEM_GUARDRAILS
            if system_prompt and system_prompt.strip():
                full_system += f"\n\nAdditional context:\n{system_prompt.strip()}"

            if context_biomarkers:
                full_system += f"\n\nVerified session ground truth:\n{json.dumps(context_biomarkers, indent=2)}"

            call_messages = [{"role": "system", "content": full_system}]
            for m in messages:
                role = m.get("role", "user")
                if role in ("user", "assistant", "system"):
                    call_messages.append({"role": role, "content": m.get("content", "")})

            res_json = await self._call_chat_completions(
                call_messages,
                temperature=0.7,
                max_tokens=800,
            )
            reply = res_json["choices"][0]["message"]["content"]
            return ChatResponse(reply=reply, is_fallback=False, model=self.model_id)

        except Exception as exc:
            logger.warning("Nebius chat call failed (%s); returning graceful fallback response.", exc)
            return ChatResponse(
                reply=self._deterministic_chat_response(messages),
                is_fallback=True,
                model=self.model_id,
            )

    @staticmethod
    def _deterministic_chat_response(messages: List[Dict[str, str]]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "").lower()
                break

        if "risk" in last_user or "tier" in last_user:
            return (
                "CogniSafe calculates risk tiers by comparing your current acoustic and linguistic speech biomarkers "
                "against your established personal historical baseline across multiple sessions. A tier is only assigned "
                "when sufficient baseline history exists (minimum 5 sessions)."
            )
        if "biomarker" in last_user or "acoustic" in last_user or "nlp" in last_user:
            return (
                "CogniSafe monitors fourteen speech biomarkers: ten acoustic features (such as speech rate, pause frequency, "
                "pitch range, jitter, shimmer, and HNR) and four linguistic features (lexical diversity, semantic coherence, "
                "idea density, and syntactic complexity)."
            )
        if "doctor" in last_user or "diagnos" in last_user or "alzheimer" in last_user or "dementia" in last_user:
            return (
                "CogniSafe is a voice-first cognitive-health monitoring platform, not a diagnostic medical device. It does "
                "not provide medical diagnoses. For clinical evaluations or medical concerns, please consult a qualified neurologist or physician."
            )
        return (
            "Hello! I am your CogniSafe AI assistant powered by NVIDIA Nemotron 3.5 Lightning. I can help explain your voice biomarker "
            "metrics, longitudinal trends, and personal baseline comparisons. How can I assist you today?"
        )


_default_nebius_client: Optional[NebiusClient] = None


def get_nebius_client() -> NebiusClient:
    """Retrieve the singleton NebiusClient instance."""
    global _default_nebius_client
    if _default_nebius_client is None:
        _default_nebius_client = NebiusClient()
    return _default_nebius_client
