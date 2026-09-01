"""
Bhashini Provider  (Phase 8)
──────────────────────────────
Bhashini is India's AI-driven language translation platform managed
by MeitY (Ministry of Electronics and Information Technology).

API: https://bhashini.gov.in/ulca/model/api-info/nmt

This module provides a production-ready integration point.
The system automatically falls back to LLMTranslationProvider
when Bhashini credentials are not configured.

Configuration (in .env):
    BHASHINI_USER_ID   = your_user_id
    BHASHINI_API_KEY   = your_api_key
    BHASHINI_PIPELINE_ID = your_pipeline_id   # optional

Usage:
    provider = BhashiniProvider()
    if provider.is_available():
        english = provider.translate_to_english("नीम और हल्दी", Language.HINDI)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from multilingual.schemas import Language, LANGUAGE_NAMES
from multilingual.provider import TranslationProvider, protect_terms, restore_terms

# Bhashini language codes (ISO 639-1 → Bhashini sourceLanguage)
_BHASHINI_CODES: dict[Language, str] = {
    Language.HINDI   : "hi",
    Language.KANNADA : "kn",
    Language.ENGLISH : "en",
}

# Bhashini ULCA pipeline endpoints
_BHASHINI_PIPELINE_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
_BHASHINI_INFERENCE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"


class BhashiniProvider(TranslationProvider):
    """
    Translation via Bhashini / ULCA NMT pipeline.

    Falls back gracefully to returning the original text (which triggers
    the caller to use LLMTranslationProvider instead).
    """

    def __init__(self):
        self._user_id     = os.environ.get("BHASHINI_USER_ID", "")
        self._api_key     = os.environ.get("BHASHINI_API_KEY", "")
        self._pipeline_id = os.environ.get("BHASHINI_PIPELINE_ID", "")
        self._session: Optional[object] = None

    def is_available(self) -> bool:
        """True only when both USER_ID and API_KEY are set."""
        return bool(self._user_id) and bool(self._api_key)

    # ──────────────────────────────────────────────────────────────────────────
    # TranslationProvider interface
    # ──────────────────────────────────────────────────────────────────────────

    def translate_to_english(self, text: str, source_language: Language) -> str:
        if self._is_english(source_language):
            return text
        return self._translate(text, source_language, Language.ENGLISH)

    def translate_from_english(self, text: str, target_language: Language) -> str:
        if self._is_english(target_language):
            return text
        protected, mapping = protect_terms(text)
        result = self._translate(protected, Language.ENGLISH, target_language)
        return restore_terms(result, mapping)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Bhashini ULCA call
    # ──────────────────────────────────────────────────────────────────────────

    def _translate(
        self,
        text: str,
        source: Language,
        target: Language,
    ) -> str:
        """
        Call the Bhashini ULCA inference pipeline.

        Returns the translated text, or raises an exception (which the
        caller should catch and fall back to LLM translation).
        """
        try:
            import urllib.request
        except ImportError:
            raise RuntimeError("urllib not available")

        src_code = _BHASHINI_CODES.get(source, "en")
        tgt_code = _BHASHINI_CODES.get(target, "en")

        # ── Step 1: Get pipeline config ────────────────────────────────────
        pipeline_payload = json.dumps({
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": src_code,
                            "targetLanguage": tgt_code,
                        }
                    }
                }
            ],
            "pipelineRequestConfig": {
                "pipelineId": self._pipeline_id or "64392f96daac500b55c543cd"
            }
        }).encode("utf-8")

        pipeline_req = urllib.request.Request(
            _BHASHINI_PIPELINE_URL,
            data=pipeline_payload,
            headers={
                "Content-Type": "application/json",
                "userID"      : self._user_id,
                "ulcaApiKey"  : self._api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(pipeline_req, timeout=10) as resp:
            pipeline_data = json.loads(resp.read().decode("utf-8"))

        # Extract service URL and model ID from response
        service_url = (
            pipeline_data
            .get("pipelineInferenceAPIEndPoint", {})
            .get("callbackUrl", _BHASHINI_INFERENCE_URL)
        )
        inference_key = (
            pipeline_data
            .get("pipelineInferenceAPIEndPoint", {})
            .get("inferenceApiKey", {})
            .get("value", self._api_key)
        )
        model_id = (
            pipeline_data
            .get("pipelineResponseConfig", [{}])[0]
            .get("config", [{}])[0]
            .get("modelId", "")
        )

        # ── Step 2: Inference ──────────────────────────────────────────────
        inference_payload = json.dumps({
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": src_code,
                            "targetLanguage": tgt_code,
                        },
                        "modelId": model_id,
                    }
                }
            ],
            "inputData": {
                "input": [{"source": text}]
            }
        }).encode("utf-8")

        inference_req = urllib.request.Request(
            service_url,
            data=inference_payload,
            headers={
                "Content-Type"  : "application/json",
                "Authorization" : inference_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(inference_req, timeout=15) as resp:
            inference_data = json.loads(resp.read().decode("utf-8"))

        translated = (
            inference_data
            .get("pipelineResponse", [{}])[0]
            .get("output", [{}])[0]
            .get("target", text)   # fall back to original if parse fails
        )
        return translated

    # ──────────────────────────────────────────────────────────────────────────
    # Health check
    # ──────────────────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if Bhashini is reachable and configured."""
        if not self.is_available():
            return False
        try:
            result = self._translate("hello", Language.ENGLISH, Language.HINDI)
            return bool(result)
        except Exception:
            return False
