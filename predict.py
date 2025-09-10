from typing import Optional, List, Dict, Any
from cog import BasePredictor, Input
import json, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig,
    StoppingCriteria, StoppingCriteriaList
)

# ---- Pydantic v1/v2 compatibility -----------------------------------------
try:
    from pydantic import BaseModel, Field, HttpUrl, ValidationError
    _IS_PYD2 = hasattr(BaseModel, "model_validate")
except Exception as _e:
    raise RuntimeError(f"Pydantic not available: {_e}")

def _validate_schema(model_cls, obj: Dict[str, Any]) -> Dict[str, Any]:
    # sanitize empty URLs -> None (both v1/v2 accept None for Optional[HttpUrl])
    for k in ("code_url", "data_url"):
        v = obj.get(k, None)
        if isinstance(v, str) and v.strip() in ("", "null", "None"):
            obj[k] = None
    if _IS_PYD2:
        parsed = model_cls.model_validate(obj)             # pydantic v2
        return parsed.model_dump(mode="json")
    else:
        parsed = model_cls.parse_obj(obj)                  # pydantic v1
        return json.loads(parsed.json())

# ---- Schema ----------------------------------------------------------------
class Benchmark(BaseModel):
    name: str
    split_or_suite: Optional[str] = None
    notes: Optional[str] = None

class Metric(BaseModel):
    name: str
    value: str
    higher_is_better: Optional[bool] = None

class Extracted(BaseModel):
    title: str
    venue_year: Optional[str] = None
    task_type: List[str]
    problem_statement: str
    method_keywords: List[str]
    robot_platforms: List[str]
    sensors: List[str] = []
    actuators: List[str] = []
    environment: str
    benchmarks: List[Benchmark] = []
    metrics: List[Metric] = []
    baselines: List[str] = []
    results_summary: List[str]
    limitations: List[str] = []
    code_url: Optional[HttpUrl] = None
    data_url: Optional[HttpUrl] = None

SYSTEM_PROMPT = (
    "You are a senior robotics reviewer. Extract JSON with fields:"
    " title, venue_year, task_type, problem_statement, method_keywords,"
    " robot_platforms, sensors, actuators, environment, benchmarks,"
    " metrics, baselines, results_summary, limitations, code_url, data_url.\n"
    "Return ONLY strict JSON. Begin with '{' and end with '}'."
)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# ---- Robust JSON carving ----------------------------------------------------
def _first_balanced_json_object(text: str, max_extra_braces: int = 3):
    if "{" not in text:
        return None
    s = text[text.find("{"):]
    def carve(txt):
        depth, start = 0, None
        for i, ch in enumerate(txt):
            if ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return txt[start:i+1]
        return None
    cand = carve(s)
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            pass
    need = s.count("{") - s.count("}")
    # try appending just enough braces (or up to 3 more) and re-parse
    for k in range(1, max(need, 0) + 1 if need > 0 else max_extra_braces + 1):
        cand = carve(s + ("}" * k))
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None

# ---- Early stop when top-level JSON closes ---------------------------------
class BalancedJSONStop(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.depth = 0
        self.started = False
    def __call__(self, input_ids, scores, **kwargs):
        # decode only the last token; cheap but effective for braces
        s = self.tok.decode(input_ids[0, -1:], skip_special_tokens=True)
        for ch in s:
            if ch == "{":
                self.started = True
                self.depth += 1
            elif ch == "}" and self.started:
                self.depth -= 1
                if self.depth == 0:
                    return True
        return False

# ---- Predictor --------------------------------------------------------------
class Predictor(BasePredictor):
    def setup(self):
        # Tokenizer (fast preferred)
        try:
            self.tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

        # GPU info
        name_mem = None
        if torch.cuda.is_available():
            d = torch.cuda.get_device_properties(0)
            name_mem = (d.name, round(d.total_memory/(1024**3), 2))

        # Model: bf16 on big GPUs, else 4-bit
        if name_mem and ("A100" in name_mem[0] or name_mem[1] >= 30):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            mdl = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
            )
            self.mode = "bf16"
        else:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else None
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb, device_map="auto"
            )
            self.mode = "4bit-NF4"

        self.pipe = pipeline("text-generation", model=mdl, tokenizer=self.tok)
        # set pad token if missing to silence warnings
        if getattr(self.pipe.model.config, "pad_token_id", None) is None:
            self.pipe.model.config.pad_token_id = self.tok.eos_token_id

        # deterministic, bounded-time generation
        self.gen_kw = dict(
            max_new_tokens=360,          # tighter cap; faster/cheaper
            do_sample=False,             # deterministic
            return_full_text=False,      # no prompt echo
            use_cache=False,             # lower VRAM
            eos_token_id=self.tok.eos_token_id,
            max_time=60,                 # HARD STOP after 60s of generation
        )
        self.stopper = StoppingCriteriaList([BalancedJSONStop(self.tok)])

    def predict(self, document: str = Input(description="Paste robotics text", default="")) -> Dict[str, Any]:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document},
        ]
        text_in = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        with torch.inference_mode():
            out = self.pipe(text_in, stopping_criteria=self.stopper, **self.gen_kw)[0]["generated_text"]

        obj = _first_balanced_json_object(out)
        if obj is None:
            return {"error": "no_json_found", "tail": out[-800:], "mode": self.mode}

        # normalize drift (bool -> str for split_or_suite)
        if isinstance(obj, dict) and isinstance(obj.get("benchmarks"), list):
            for b in obj["benchmarks"]:
                if isinstance(b, dict) and isinstance(b.get("split_or_suite"), bool):
                    b["split_or_suite"] = "true" if b["split_or_suite"] else "false"

        try:
            return _validate_schema(Extracted, obj)
        except ValidationError as e:
            return {"error": "schema_validation_failed", "detail": str(e)[:600], "raw": obj, "mode": self.mode}
