"""
CogniSafe NLP Biomarkers Pipeline (4 Linguistic/NLP Biomarkers).
Implements standard MTLD (retaining stopwords), consecutive + global sentence embedding
coherence, CPIDR proposition heuristic excluding auxiliary verbs, and recursion-safe
iterative syntactic parse tree traversal with depth guards.
"""

from collections import deque
import logging
from typing import Optional, Dict, Any, Union, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer

from pipeline.schemas import (
    FeatureStatus,
    TypedFeatureResult,
    BiomarkerCollectionResult,
    CANONICAL_NLP_BIOMARKERS,
)

logger = logging.getLogger("cognisafe.nlp")

# Lazy model singletons
_NLP = None
_EMBEDDER = None


def get_spacy_model():
    """Lazy load spaCy model without crashing Windows cp1252 consoles."""
    global _NLP
    if _NLP is None:
        try:
            logger.info("[NLP] Loading spaCy model 'en_core_web_sm'...")
            _NLP = spacy.load("en_core_web_sm")
            logger.info("[NLP] spaCy model loaded [OK]")
        except Exception as e:
            logger.error(f"[NLP ERROR] Failed to load spaCy model: {e}")
            raise
    return _NLP


def get_embedder():
    """Lazy load sentence transformer model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            logger.info("[NLP] Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[NLP] SentenceTransformer loaded [OK]")
        except Exception as e:
            logger.error(f"[NLP ERROR] Failed to load SentenceTransformer: {e}")
            raise
    return _EMBEDDER


def analyze_text(
    transcript_data: Union[dict, str],
    return_typed: bool = True,
) -> Union[BiomarkerCollectionResult, Dict[str, Any]]:
    """
    Analyze transcript text and extract the 4 canonical NLP biomarkers:
    1. lexical_diversity (MTLD score retaining all alpha tokens, minimum 20 tokens)
    2. semantic_coherence (cosine similarity between consecutive & global sentences)
    3. idea_density (propositions / total words, excluding auxiliary verbs & particles)
    4. syntactic_complexity (average parse tree depth via iterative traversal with cycle guard)

    Eliminates all silent 0.0 fallbacks. Missing or insufficient text inputs
    return value: None with explicit FeatureStatus.
    """
    # ── Step 0: Extract text safely ───────────────────────────────────────────
    if isinstance(transcript_data, dict):
        text = transcript_data.get("text", "")
    elif isinstance(transcript_data, str):
        text = transcript_data
    else:
        text = ""

    text = text.strip() if text else ""

    if not text:
        return _insufficient_nlp_result(reason="empty_or_missing_transcript", return_typed=return_typed)

    try:
        nlp = get_spacy_model()
        doc = nlp(text)

        # ── 1. Lexical Diversity (Standard MTLD) ──────────────────────────────
        lexical_diversity = compute_mtld(doc)

        # ── 2. Semantic Coherence (Sentence Embeddings) ────────────────────────
        semantic_coherence = compute_semantic_coherence(doc, text)

        # ── 3. Idea Density (CPIDR Proposition Heuristic) ─────────────────────
        idea_density = compute_idea_density(doc)

        # ── 4. Syntactic Complexity (Iterative Parse Depth) ───────────────────
        syntactic_complexity = compute_syntactic_complexity(doc)

        features = {
            "lexical_diversity": lexical_diversity,
            "semantic_coherence": semantic_coherence,
            "idea_density": idea_density,
            "syntactic_complexity": syntactic_complexity,
        }

        collection = BiomarkerCollectionResult(
            features=features,
            overall_status="complete",
            metadata={"character_length": len(text), "token_count": len(doc)}
        )
        return collection if return_typed else collection.to_dict()

    except Exception as e:
        logger.error(f"[NLP ERROR] Text analysis failed: {e}", exc_info=True)
        return _failed_nlp_result(reason=str(e), return_typed=return_typed)


# ── MTLD — Measure of Textual Lexical Diversity ──────────────────────────────
def compute_mtld(doc, threshold: float = 0.72) -> TypedFeatureResult:
    """
    Compute standard bidirectional MTLD (McCarthy & Jarvis, 2010).
    Critical correction: Retains ALL alpha tokens (content + function words/stopwords)
    as required by standard MTLD methodology.
    Requires at least 20 tokens to provide a valid estimate; otherwise reports insufficient_input.
    """
    tokens = [token.text.lower() for token in doc if token.is_alpha]

    if len(tokens) < 20:
        return TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="MTLD",
            is_proxy=False,
            method="standard_bidirectional_mtld",
            reason="insufficient_tokens_min_20",
            metadata={"token_count": len(tokens), "min_required": 20}
        )

    def mtld_unidirectional_pass(token_sequence: List[str]) -> float:
        factor_count = 0.0
        token_count = 0
        types = set()

        for token in token_sequence:
            token_count += 1
            types.add(token)
            ttr = len(types) / float(token_count)

            if ttr <= threshold:
                factor_count += 1.0
                token_count = 0
                types.clear()

        # Partial factor calculation for trailing tokens
        if token_count > 0:
            ttr = len(types) / float(token_count)
            if ttr < 1.0:
                factor_count += (1.0 - ttr) / (1.0 - threshold)

        return len(token_sequence) / factor_count if factor_count > 0 else float(len(token_sequence))

    forward = mtld_unidirectional_pass(tokens)
    backward = mtld_unidirectional_pass(list(reversed(tokens)))
    mtld_score = round(float((forward + backward) / 2.0), 4)

    return TypedFeatureResult(
        status=FeatureStatus.VALID,
        value=mtld_score,
        unit="MTLD",
        is_proxy=False,
        method="standard_bidirectional_mtld",
        metadata={
            "token_count": len(tokens),
            "unique_types": len(set(tokens)),
            "forward_score": round(forward, 4),
            "backward_score": round(backward, 4),
        }
    )


# ── Semantic Coherence ────────────────────────────────────────────────────────
def compute_semantic_coherence(doc, text: str) -> TypedFeatureResult:
    """
    Measures semantic connectedness across sentences using SentenceTransformer embeddings.
    Combines consecutive sentence cosine similarity and global centroid similarity.
    Requires at least 2 sentences; single-sentence or empty transcripts return value: None.
    """
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) >= 3]

    if len(sentences) < 2:
        return TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="cosine_similarity",
            is_proxy=False,
            method="consecutive_and_global_sentence_embedding",
            reason="fewer_than_2_sentences",
            metadata={"sentence_count": len(sentences), "min_required": 2}
        )

    try:
        embedder = get_embedder()
        embeddings = embedder.encode(sentences)

        # 1. Consecutive sentence similarities
        consec_sims = []
        for i in range(1, len(embeddings)):
            sim = float(
                cosine_similarity(
                    embeddings[i - 1].reshape(1, -1),
                    embeddings[i].reshape(1, -1)
                )[0][0]
            )
            consec_sims.append(sim)
        local_coherence = float(np.mean(consec_sims)) if consec_sims else 0.0

        # 2. Global centroid similarity (sentence to topic centroid)
        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        global_sims = [
            float(cosine_similarity(emb.reshape(1, -1), centroid)[0][0])
            for emb in embeddings
        ]
        global_coherence = float(np.mean(global_sims)) if global_sims else 0.0

        # Weighted composite: 50% local sequential + 50% global thematic
        composite = 0.5 * local_coherence + 0.5 * global_coherence
        bounded_score = round(float(max(0.0, min(1.0, composite))), 4)

        return TypedFeatureResult(
            status=FeatureStatus.VALID,
            value=bounded_score,
            unit="cosine_similarity",
            is_proxy=False,
            method="consecutive_and_global_sentence_embedding",
            metadata={
                "sentence_count": len(sentences),
                "local_coherence": round(local_coherence, 4),
                "global_coherence": round(global_coherence, 4),
            }
        )
    except Exception as e:
        logger.error(f"[NLP ERROR] Semantic coherence failed: {e}")
        return TypedFeatureResult(
            status=FeatureStatus.EXTRACTION_FAILED,
            value=None,
            unit="cosine_similarity",
            is_proxy=False,
            method="consecutive_and_global_sentence_embedding",
            reason=str(e)
        )


# ── Idea Density ──────────────────────────────────────────────────────────────
def compute_idea_density(doc) -> TypedFeatureResult:
    """
    Propositional idea density heuristic approximating CPIDR.
    Extracts informative propositions (main verbs, adjectives, adverbs, prepositions)
    while strictly excluding auxiliary verbs ('is', 'have', 'do' when auxiliary),
    particles ('to', 'not'), and punctuation.
    """
    alpha_tokens = [token for token in doc if token.is_alpha]
    total_words = len(alpha_tokens)

    if total_words < 10:
        return TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="ratio",
            is_proxy=True,
            method="cpidr_proposition_pos_heuristic_excluding_auxiliaries",
            reason="fewer_than_10_words",
            metadata={"total_words": total_words, "min_required": 10}
        )

    propositions = []
    for token in alpha_tokens:
        pos = token.pos_
        dep = token.dep_

        # Exclude auxiliary verbs and modals
        if pos == "AUX" or dep in ("aux", "auxpass"):
            continue

        # Exclude particles
        if pos == "PART":
            continue

        # Valid propositions in CPIDR:
        # - Main verbs (actions, states)
        # - Adjectives (descriptors)
        # - Adverbs (modifiers)
        # - Prepositions & subordinating conjunctions (relational connectives)
        if pos in ("VERB", "ADJ", "ADV", "ADP", "SCONJ"):
            propositions.append(token.text.lower())

    density = len(propositions) / float(total_words)
    bounded_density = round(float(max(0.0, min(1.0, density))), 4)

    return TypedFeatureResult(
        status=FeatureStatus.VALID,
        value=bounded_density,
        unit="ratio",
        is_proxy=True,
        method="cpidr_proposition_pos_heuristic_excluding_auxiliaries",
        metadata={
            "proposition_count": len(propositions),
            "total_words": total_words,
            "sample_propositions": propositions[:10]
        }
    )


# ── Syntactic Complexity ──────────────────────────────────────────────────────
def compute_syntactic_complexity(doc, max_depth_guard: int = 20) -> TypedFeatureResult:
    """
    Average dependency parse tree depth across sentences.
    Uses iterative BFS traversal with cycle detection and a hard depth guard
    (max_depth=20) to guarantee recursion safety on malformed or repetitive text.
    """
    sentence_depths = []

    for sent in doc.sents:
        roots = [token for token in sent if token.dep_ == "ROOT"]
        if not roots:
            continue

        root = roots[0]
        # Iterative BFS traversal to prevent RecursionError on malformed trees
        visited = {root.i}
        queue = deque([(root, 1)])
        max_depth_for_sentence = 1

        while queue:
            node, curr_depth = queue.popleft()
            if curr_depth > max_depth_for_sentence:
                max_depth_for_sentence = min(curr_depth, max_depth_guard)

            if curr_depth < max_depth_guard:
                for child in node.children:
                    if child.i not in visited:
                        visited.add(child.i)
                        queue.append((child, curr_depth + 1))

        sentence_depths.append(max_depth_for_sentence)

    if not sentence_depths:
        return TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="depth",
            is_proxy=False,
            method="iterative_dep_parse_tree_depth",
            reason="no_valid_parse_trees"
        )

    mean_depth = round(float(np.mean(sentence_depths)), 4)
    return TypedFeatureResult(
        status=FeatureStatus.VALID,
        value=mean_depth,
        unit="depth",
        is_proxy=False,
        method="iterative_dep_parse_tree_depth",
        metadata={
            "sentence_count": len(sentence_depths),
            "sentence_depths": sentence_depths,
            "max_depth_guard": max_depth_guard
        }
    )


def _insufficient_nlp_result(reason: str, return_typed: bool):
    units = {
        "lexical_diversity": "MTLD",
        "semantic_coherence": "cosine_similarity",
        "idea_density": "ratio",
        "syntactic_complexity": "depth",
    }
    is_proxies = {
        "lexical_diversity": False,
        "semantic_coherence": False,
        "idea_density": True,
        "syntactic_complexity": False,
    }
    methods = {
        "lexical_diversity": "standard_bidirectional_mtld",
        "semantic_coherence": "consecutive_and_global_sentence_embedding",
        "idea_density": "cpidr_proposition_pos_heuristic_excluding_auxiliaries",
        "syntactic_complexity": "iterative_dep_parse_tree_depth",
    }
    features = {
        name: TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit=units[name],
            is_proxy=is_proxies[name],
            method=methods[name],
            reason=reason,
        )
        for name in CANONICAL_NLP_BIOMARKERS
    }
    collection = BiomarkerCollectionResult(
        features=features,
        overall_status="insufficient_data",
        metadata={"reason": reason}
    )
    return collection if return_typed else collection.to_dict()


def _failed_nlp_result(reason: str, return_typed: bool):
    units = {
        "lexical_diversity": "MTLD",
        "semantic_coherence": "cosine_similarity",
        "idea_density": "ratio",
        "syntactic_complexity": "depth",
    }
    is_proxies = {
        "lexical_diversity": False,
        "semantic_coherence": False,
        "idea_density": True,
        "syntactic_complexity": False,
    }
    methods = {
        "lexical_diversity": "standard_bidirectional_mtld",
        "semantic_coherence": "consecutive_and_global_sentence_embedding",
        "idea_density": "cpidr_proposition_pos_heuristic_excluding_auxiliaries",
        "syntactic_complexity": "iterative_dep_parse_tree_depth",
    }
    features = {
        name: TypedFeatureResult(
            status=FeatureStatus.EXTRACTION_FAILED,
            value=None,
            unit=units[name],
            is_proxy=is_proxies[name],
            method=methods[name],
            reason=reason,
        )
        for name in CANONICAL_NLP_BIOMARKERS
    }
    collection = BiomarkerCollectionResult(
        features=features,
        overall_status="failed",
        metadata={"error": reason}
    )
    return collection if return_typed else collection.to_dict()