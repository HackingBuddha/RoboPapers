# predict.py — RoboPapers (robust JSON, early stop, pydantic v2)

from typing import Optional, List, Dict, Any
from cog import BasePredictor, Input
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig,
    StoppingCriteria, StoppingCriteriaList
)
import torch, json, os

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")

# ---------------- Schema ----------------
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

# ---------------- Prompt (tight length budget) ----------------
SYSTEM_PROMPT = """You are a senior robotics reviewer.
Return ONLY JSON matching the schema below. Begin with '{' and end with '}'. Output nothing else.

Rules (hard):
- Keep each string ≤ 120 characters.
- results_summary: 3–5 bullets, each must include a number.
- If unknown: use [] or null. No guesses. No prose outside JSON.
- Total output ≤ 1800 characters.

Schema fields:
- title
- venue_year (ICRA/IROS/RSS/CoRL year or arXiv year if given)
- task_type: list from {manipulation, navigation, locomotion, aerial, mobile manipulation, multi-robot, planning/control}
- problem_statement: one sentence
- method_keywords: 3–6 (e.g., diffusion policy, MPC, RL, imitation, LfD, visuomotor transformer)
- robot_platforms: list (Franka, UR5, ANYmal, quadrotor, TurtleBot, etc.)
- sensors: list (RGB, depth, LiDAR, proprioception, F/T, tactile)
- actuators: list
- environment: 'sim', 'real', or 'sim-to-real'; include simulator if named (Isaac, Mujoco, Habitat, Gibson)
- benchmarks: array of {name, split_or_suite?, notes?}
- metrics: array of {name, value, higher_is_better?}
- baselines: list
- results_summary: 3–5 bullets
- limitations: 1–4 bullets
- code_url, data_url: if present
"""

# ---------------- Robust JSON slicer/repair ----------------
def _first_balanced_json_object(text: str, max_extra_braces: int = 64):
    """Carve FIRST balanced top-level {...}; if needed, append up to N '}' to balance."""
    if "{" not in text:
        return None
    s = text[text.find("{"):]  # drop any preface
    def carve(s2):
        depth, start = 0, None
        for i, ch in enumerate(s2):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return s2[start:i+1]
        return None
    # try as-is
    cand = carve(s)
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            pass
    # try with added braces
    need = s.count("{") - s.count("}")
    limit = max(need, 0) if need > 0 else max_extra_braces
    for k in range(1, limit + 1):
        cand = carve(s + ("}" * k))
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None

def _coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize bool -> str for split_or_suite
    if isinstance(d, dict) and isinstance(d.get("benchmarks"), list):
        for b in d["benchmarks"]:
            if isinstance(b, dict) and isinstance(b.get("split_or_suite"), bool):
                b["split_or_suite"] = "true" if b["split_or_suite"] else "false"
    # Empty URL strings → None (pydantic HttpUrl-safe)
    for k in ("code_url", "data_url"):
        v = d.get(k)
        if isinstance(v, str) and not v.strip():
            d[k] = None
    return d

class BalancedJSONStop(StoppingCriteria):
    """Stop generation the moment the top-level JSON object closes."""
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.depth = 0
        self.started = False
    def __call__(self, input_ids, scores, **kwargs):
        # decode only the last token; works well for Qwen3 tokenization
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

# ---------------- Predictor ----------------
class Predictor(BasePredictor):
    def setup(self):
        # Tokenizer (prefer fast)
        try:
            self.tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

        # Prefer bf16 on big GPUs; else 4-bit NF4
        name_mem = None
        if torch.cuda.is_available():
            d = torch.cuda.get_device_properties(0)
            name_mem = (d.name, round(d.total_memory/(1024**3), 2))
        if name_mem and ("A100" in name_mem[0] or name_mem[1] >= 30):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            mdl = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
            )
            self.mode = "bf16"
        else:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb, device_map="auto"
            )
            self.mode = "4bit-NF4"

        self.pipe = pipeline("text-generation", model=mdl, tokenizer=self.tok)
        if getattr(self.pipe.model.config, "pad_token_id", None) is None:
            self.pipe.model.config.pad_token_id = self.tok.eos_token_id

        # Deterministic, short budget; stop when JSON closes
        self.kw = dict(
            max_new_tokens=320,     # smaller cap → faster & cheaper
            do_sample=False,
            return_full_text=False,
            use_cache=False,
            eos_token_id=self.tok.eos_token_id,
            # You can also set max_time to hard-cap a runaway output:
            # max_time=110,
        )
        self.stop = StoppingCriteriaList([BalancedJSONStop(self.tok)])

    def predict(self,
                document: str = Input(description="Paste robotics text", default="")) -> Dict[str, Any]:

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document},
        ]
        text_in = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        with torch.inference_mode():
            out = self.pipe(text_in, stopping_criteria=self.stop, **self.kw)[0]["generated_text"]

        obj = _first_balanced_json_object(out)
        if obj is None:
            # last-ditch: return tail for debugging instead of failing hard
            return {"error": "no_json_found", "mode": self.mode, "tail": out[-800:]}

        obj = _coerce_types(obj)
        try:
            parsed = Extracted.model_validate(obj)         # pydantic v2
            return parsed.model_dump(mode="json")          # JSON-safe (HttpUrl → str)
        except ValidationError as e:
            return {"error": "schema_validation_failed", "detail": str(e)[:600], "raw": obj}
