import json
import re
import sys
from typing import List, Optional, Any, Dict

import torch
from cog import BasePredictor, Input
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ---------------------------
# Pydantic v1/v2 compatibility
# ---------------------------
try:
    # v2
    from pydantic import BaseModel, Field, ValidationError
    _HAS_V2 = True
except Exception:  # pragma: no cover
    # v1 fallback
    from pydantic.v1 import BaseModel, Field, ValidationError  # type: ignore
    _HAS_V2 = False


def _validate(model_cls, data):
    """Use v2 .model_validate if present, else v1 .parse_obj."""
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)  # pydantic v2
    return model_cls.parse_obj(data)           # pydantic v1


def _dump(model_obj) -> Dict[str, Any]:
    """Use v2 .model_dump if present, else v1 .dict."""
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


# ---------------------------
# Output schema
# ---------------------------
class Extracted(BaseModel):
    # keep URLs as plain strings to avoid strict validators rejecting good data
    title: str = Field("", description="Paper title")
    venue_year: str = Field("", description="Conference/journal + year, or year only")
    task_type: str = Field("", description="Core task area, e.g. navigation, manipulation")
    problem_statement: str = Field("", description="One-sentence problem framing")
    method_keywords: List[str] = Field(default_factory=list)
    robot_platforms: List[str] = Field(default_factory=list)
    sensors: List[str] = Field(default_factory=list)
    actuators: List[str] = Field(default_factory=list)
    environment: str = Field("", description="Operating environment")
    benchmarks: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    baselines: List[str] = Field(default_factory=list)
    dataset_names: List[str] = Field(default_factory=list)
    code_url: Optional[str] = Field(None, description="Repo URL if present")
    data_url: Optional[str] = Field(None, description="Dataset URL if present")


# ---------------------------
# Small helpers
# ---------------------------
_SYSTEM_PROMPT = (
    "You extract structured fields from short robotics abstracts. "
    "Return **one JSON object only** with the keys of the schema I provide. "
    "Do not add prose before or after the JSON. If a field is unknown, use an empty "
    "string or empty list appropriately. Key names must match exactly."
)

_SCHEMA_EXAMPLE = {
    "title": "Example Title",
    "venue_year": "ICRA 2024",
    "task_type": "navigation",
    "problem_statement": "One sentence.",
    "method_keywords": ["keyword1", "keyword2"],
    "robot_platforms": [],
    "sensors": [],
    "actuators": [],
    "environment": "",
    "benchmarks": [],
    "metrics": [],
    "baselines": [],
    "dataset_names": [],
    "code_url": "",
    "data_url": ""
}


def _strip_code_fences(text: str) -> str:
    # remove ```json ... ``` wrappers if present
    return re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())


def _first_balanced_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Find and parse the first balanced {...} JSON object in text.
    Handles extra tokens around JSON, code fences, and nested braces inside strings.
    """
    s = _strip_code_fences(text)

    # Fast path: whole string is JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # General scan
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break  # try next '{'
        start = s.find("{", start + 1)

    return None


def _sanitize_urls(d: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("code_url", "data_url"):
        v = d.get(k)
        if isinstance(v, str) and v:
            v = v.strip()
            # very light cleanup; keep only plausible http(s) links
            if not re.match(r"^https?://", v):
                v = ""
            d[k] = v
    return d


# ---------------------------
# Predictor
# ---------------------------
class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Load the model once when the container starts.

        We try bf16 first (preferred on modern NVIDIA GPUs).
        If memory is tight, users can drop to 4-bit NF4 by setting `quant_4bit=True`.
        """
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"  # change if you prefer Qwen3-4B
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        # Default path: bfloat16 (fast and stable)
        self._quant_4bit_default = False
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map="auto",
        )

    def _load_4bit(self):
        """Lazy-switch to 4-bit if requested at runtime."""
        global torch
        from transformers import BitsAndBytesConfig  # imported lazily
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            ),
        )

    def predict(
        self,
        document: str = Input(description="Short robotics abstract text"),
        quant_4bit: bool = Input(
            description="Use 4-bit NF4 quantization (saves VRAM; a bit slower vs bf16).",
            default=False,
        ),
        max_new_tokens: int = Input(
            description="Generation cap; 512 is plenty for one JSON object.",
            default=384,
            ge=64,
            le=1024,
        ),
    ) -> Dict[str, Any]:
        # Optional runtime switch to 4-bit
        if quant_4bit and (not self._quant_4bit_default):
            self._load_4bit()
            self._quant_4bit_default = True

        tok = self.tokenizer

        # Chat-style prompt -> raw text with the model's template
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Schema (example keys): {json.dumps(_SCHEMA_EXAMPLE)}"},
            {"role": "user", "content": f"Abstract:\n{document.strip()}"},
            {"role": "user", "content": "Return one JSON object only. No extra text."},
        ]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Deterministic decode: no temperature/top_p/top_k
        inputs = tok(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        out_text = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Try to parse JSON robustly
        obj = _first_balanced_json(out_text)
        if obj is None:
            # Graceful failure with tail for debugging
            tail = _strip_code_fences(out_text)
            return {
                "error": "no_json_found",
                "mode": "4bit-NF4" if self._quant_4bit_default else "bf16",
                "tail": tail[:800],
            }

        # Light cleanup & validation
        obj = _sanitize_urls(obj)
        try:
            parsed = _validate(Extracted, obj)
        except ValidationError as e:
            return {
                "error": "validation_failed",
                "mode": "4bit-NF4" if self._quant_4bit_default else "bf16",
                "message": str(e)[:800],
                "raw": obj,
            }

        clean = _dump(parsed)
        return clean
