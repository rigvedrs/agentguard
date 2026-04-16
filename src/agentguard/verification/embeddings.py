"""Embedding-based semantic similarity for tool output verification.

Optional module — requires ``sentence-transformers`` to be installed.
If not available, all functions return neutral scores.

Research basis:
- Section 7.4, Novel Contribution 2: "Composite embedding fingerprint"
  Combine (query embedding, tool output embedding, argument embedding) into
  a three-way similarity check. A hallucinated result will often have low
  coherence between all three.
- Section 4.1: Embedding similarity vs tool centroid (Tier 1 signal).
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

# Soft dependency
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_model = None
_model_name = "all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def _get_model():
    """Lazy-load the sentence transformer model."""
    global _model
    if _model is None and _SENTENCE_TRANSFORMERS_AVAILABLE:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = SentenceTransformer(_model_name)
    return _model


def embed(text: str) -> Optional[list[float]]:
    """Embed a string into a vector. Returns None if sentence-transformers unavailable.

    Args:
        text: Text to embed.

    Returns:
        List of floats (embedding), or None.
    """
    model = _get_model()
    if model is None:
        return None
    vec = model.encode(text, convert_to_numpy=True)
    return vec.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Similarity in [-1, 1].
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_semantic_similarity(
    user_query: str,
    tool_result: Any,
    tool_name: str,
    centroid: Optional[list[float]] = None,
    threshold: float = 0.3,
) -> tuple[bool, float, str]:
    """Check semantic similarity between the user query and tool output.

    If sentence-transformers is not available, returns a neutral (no-signal) result.

    Args:
        user_query: The original user query that triggered the tool call.
        tool_result: The result returned by the tool.
        tool_name: Name of the tool (used for logging).
        centroid: Optional pre-computed centroid embedding for this tool's outputs.
        threshold: Minimum similarity required to not flag.

    Returns:
        (fired, score, detail) — score is probability of hallucination (0=real, 1=fake).
    """
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return False, 0.0, "sentence-transformers not available; skipping embedding check"

    # Serialise result to text for embedding
    try:
        result_text = json.dumps(tool_result, default=str)
    except Exception:
        result_text = str(tool_result)

    query_vec = embed(user_query)
    result_vec = embed(result_text)

    if query_vec is None or result_vec is None:
        return False, 0.0, "Embedding failed; skipping"

    sim = cosine_similarity(query_vec, result_vec)

    if sim < threshold:
        # Low similarity to query suggests potential hallucination
        score = (threshold - sim) / threshold  # Higher when more dissimilar
        score = min(1.0, score * 2.0)
        return True, score, (
            f"Semantic similarity between query and {tool_name} result is low: "
            f"{sim:.3f} (threshold={threshold})"
        )

    # Check against centroid if provided
    if centroid is not None:
        centroid_sim = cosine_similarity(result_vec, centroid)
        if centroid_sim < threshold:
            score = (threshold - centroid_sim) / threshold
            score = min(1.0, score * 1.5)
            return True, score, (
                f"{tool_name} result is far from expected output centroid: "
                f"similarity={centroid_sim:.3f} (threshold={threshold})"
            )

    return False, 0.0, f"Semantic similarity OK: {sim:.3f}"


def compute_centroid(embeddings: list[list[float]]) -> Optional[list[float]]:
    """Compute the centroid (mean) of a list of embeddings.

    Args:
        embeddings: List of embedding vectors (must be same dimension).

    Returns:
        Centroid vector, or None if list is empty.
    """
    if not embeddings:
        return None
    dim = len(embeddings[0])
    centroid = [0.0] * dim
    for vec in embeddings:
        for i, v in enumerate(vec):
            centroid[i] += v
    n = len(embeddings)
    return [c / n for c in centroid]


def is_available() -> bool:
    """Return True if sentence-transformers is installed and usable."""
    return _SENTENCE_TRANSFORMERS_AVAILABLE
