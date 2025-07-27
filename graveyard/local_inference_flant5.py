import logging
import json
import os
import re
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class Composer:
    @staticmethod
    def build_field_prompt(fields):
        lines = ["Fields:"]
        for f in fields:
            line = f"- {f.path} ({f.field_type}"
            if getattr(f, "required", False):
                line += ", required"
            line += ")"
            if getattr(f, "description", None):
                line += f": {f.description}"
            lines.append(line)
            if getattr(f, "pattern", None):
                lines.append(f"  Pattern: {f.pattern}")
            if getattr(f, "enum_values", None):
                lines.append(f"  Allowed values: {', '.join(f.enum_values)}")
            if getattr(f, "examples", None):
                lines.append(f"  Examples: {', '.join(f.examples)}")
        return "\n".join(lines)

    @staticmethod
    def flan_local_infer(group, segments):
        logger = logging.getLogger("composer.flan_local_infer")

        def _clean_json_like(text):
            text = text.strip()
            text = re.sub(r"(\w+):", r'"\1":', text)  # unquoted keys
            text = text.replace("'", '"')  # single to double quotes
            return text

        if not hasattr(Composer, "_flan_pipe"):
            model_name = "google/flan-t5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            Composer._flan_pipe = pipeline(
                "text2text-generation", model=model, tokenizer=tokenizer
            )

        pipe = Composer._flan_pipe

        field_info = Composer.build_field_prompt(group.fields)
        context = segments[0].text if segments else ""
        return_format = (
            "{\n  " + ",\n  ".join([f'"{f.path}": ...' for f in group.fields]) + "\n}"
        )
        prompt = (
            f"You are an expert field extractor. Given the following fields and a text, extract values and return valid JSON.\n\n"
            f"{field_info}\n\n"
            f'Text:\n"""{context}"""\n\n'
            f"Return format:\n{return_format}"
        )

        logger.info(f"FLAN-T5 Input Prompt:\n{prompt}")
        pipe_output = pipe(prompt, max_new_tokens=128)

        output_text = ""
        if (
            pipe_output
            and isinstance(pipe_output, list)
            and "generated_text" in pipe_output[0]
        ):
            output_text = _clean_json_like(pipe_output[0]["generated_text"])

        logger.info(f"FLAN-T5 Output Text:\n{output_text}")

        try:
            result_dict = json.loads(output_text)
        except Exception:
            logger.warning(f"FLAN-T5 output could not be parsed as JSON: {output_text}")
            # fallback if parsing fails
            return {
                f.path: {
                    "value": None,
                    "confidence": 0.0,
                    "source_chunks": [segments[0].id] if segments else [],
                }
                for f in group.fields
            }

        final = {}
        for f in group.fields:
            val = result_dict.get(f.path, None)
            conf = 0.85 if val else 0.0
            final[f.path] = {
                "value": val,
                "confidence": conf,
                "source_chunks": [segments[0].id] if segments else [],
            }

        return final
