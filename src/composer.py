# src/composer.py
"""
Composer Module

This module composes schema field values from Retriever results using LLM extraction and confidence scoring.
It uses OpenAI's API to extract values for schema fields and applies a confidence threshold to determine
if a field needs review. The results are returned as a nested JSON structure based on the schema.
"""

import logging
import math
import json
import tiktoken
import re

from openai import OpenAI
from typing import Dict, Any

from config import CONFIDENCE_THRESHOLD, OPENAI_API_KEY, OPENAI_MODEL


class Composer:
    """
    Composes schema field values from Retriever results using LLM extraction and confidence scoring.
    """

    def __init__(self, retriever, groups, llm_infer_fn=None, confidence_threshold=0.5):
        self.retriever = retriever
        self.groups = groups
        self.llm_infer_fn = llm_infer_fn or self.openai_infer
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        self.logger = logging.getLogger("composer")
        self._num_llm_calls = 0
        self._total_llm_tokens = 0

        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def compute_stats(self):
        return {
            "num_llm_calls": self._num_llm_calls,
            "total_llm_tokens": self._total_llm_tokens,
        }

    def compose(self, k=5) -> Dict[str, Any]:
        flat_results = {}
        field_types = {}
        self._num_llm_calls = 0
        self._total_llm_tokens = 0

        for group in self.groups:
            query_emb = (
                self.retriever.faiss.embeddings[0]
                if hasattr(self.retriever, "faiss")
                and self.retriever.faiss.index is not None
                and self.retriever.faiss.embeddings is not None
                else None
            )

            segments = self.retriever.retrieve(
                group, k=k, mode="hybrid", query_embedding=query_emb
            )

            self._num_llm_calls += 1
            self._total_llm_tokens += sum(
                len(self._encoding.encode(seg.text)) for seg in segments
            )

            field_results = self.llm_infer_fn(group, segments)

            for field in group.fields:
                path = field.path
                result = field_results.get(path, {})
                value = result.get("value")
                confidence = result.get("confidence", 0.0)
                needs_review = value is None or confidence < self.confidence_threshold

                flat_results[path] = {
                    "value": value,
                    "needs_review": needs_review,
                }

                field_types[path] = field.field_type

                self.logger.info(
                    f"Field '{path}': value={result.get('value')}, "
                    f"confidence={confidence:.2f}, review={needs_review}"
                )

        return self._compose_nested_json(flat_results, field_types)

    def _compose_nested_json(
        self, flat_results: Dict[str, Any], field_types: Dict[str, str]
    ) -> Dict[str, Any]:
        root = {}
        for field_path, result in flat_results.items():
            parts = field_path.replace("[]", ".[]").split(".")
            curr = root
            for i, part in enumerate(parts):
                is_last = i == len(parts) - 1

                if part == "[]":
                    next_part = parts[i + 1] if not is_last else None
                    if isinstance(curr, list):
                        if not curr or not isinstance(curr[-1], dict):
                            curr.append({})
                        curr = curr[-1]
                    else:
                        if next_part and next_part not in curr:
                            curr[next_part] = []
                        curr = curr.get(next_part, curr)
                elif is_last:
                    if isinstance(curr, list):
                        curr.append(result)
                    else:
                        curr[part] = result
                else:
                    if isinstance(curr, list):
                        if not curr or not isinstance(curr[-1], dict):
                            curr.append({})
                        curr = curr[-1]
                    if part not in curr:
                        next_part = parts[i + 1]
                        curr[part] = [] if next_part == "[]" else {}
                    curr = curr[part]
        return root

    @staticmethod
    def build_field_prompt(group, segments):
        """
        Build a complete, context-rich prompt for OpenAI inference for a group.
        """
        lines = [
            "Extract values for the following fields from the document. Return only a valid flat JSON object. Use exactly the field names provided. Do not convert them to arrays or objects, No nesting.",
            "",
            "Fields:",
        ]
        for f in group.fields:
            line = f'"{f.path}": {getattr(f, "field_type", "string")}'
            desc = getattr(f, "description", None)
            if desc:
                line += f" — {desc}"
            lines.append(line)
        lines.append("")
        lines.append("Document:")
        for seg in segments:
            lines.append(seg.text)
        lines.append("")
        lines.append("Expected output format:")
        lines.append("{")
        for f in group.fields:
            lines.append(f'  "{f.path}": <value or null>,')
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def openai_infer(group, segments):
        """
        Extract all field values for a group using a single prompt (flat JSON),
        and compute a shared confidence score based on logprobs.
        """
        logger = logging.getLogger("composer.openai_infer")
        results = {}
        document = "\n".join(f"- {seg.text}" for seg in segments)
        seg_ids = [seg.id for seg in segments]

        prompt = Composer.build_field_prompt(group, segments)

        client = OpenAI(api_key=OPENAI_API_KEY)

        try:
            response = client.completions.create(
                model=OPENAI_MODEL,
                prompt=prompt,
                max_tokens=128,
                temperature=0,
                logprobs=5,
                # stop=["\n}"],  # stop at the end of JSON object
                stop=None,  # let it finish naturally
            )

            output_text = response.choices[0].text.strip()

            # Try to parse JSON safely
            try:
                json_start = output_text.find("{")
                json_end = output_text.rfind("}") + 1
                json_str = output_text[json_start:json_end]

                if not json_str.strip().endswith("}"):
                    json_str += "}"  # Patch the missing brace if ending on }

                # Fix trailing comma issue
                # Remove any trailing comma before the closing brace
                json_str = re.sub(r",\s*}", "}", json_str)

                result_dict = json.loads(json_str)
            except Exception as e:
                logger.warning(
                    f" Failed to parse JSON: {e}\nRaw LLM output:\n{output_text}"
                )
                result_dict = {}

            # Compute shared confidence from entire completion logprobs
            logprobs = response.choices[0].logprobs
            if logprobs and logprobs.token_logprobs:
                avg_logprob = sum(logprobs.token_logprobs) / len(
                    logprobs.token_logprobs
                )
                confidence = round(math.exp(avg_logprob), 4)
            else:
                confidence = 0.0

        except Exception as e:
            result_dict = {}
            confidence = 0.0

        # Apply the shared confidence to each field
        for field in group.fields:
            value = result_dict.get(field.path, None)
            if isinstance(value, str) and value.lower() == "null":
                value = None

            results[field.path] = {"value": value, "confidence": confidence}

        print(
            f"Results for group {group.group_id}: {result_dict}, confidence={confidence:.2f}"
        )

        return results
