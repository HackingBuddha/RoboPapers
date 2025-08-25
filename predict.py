from typing import Optional, List, Dict, Any
from cog import BasePredictor, Input
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch, json

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# ---- Schema (same as Colab) ----
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

SYSTEM_PROMPT = """You are a senior robotics reviewer. Given a robotics paper's text, extract a JSON object with:
- title
- venue_year (ICRA/IROS/RSS/CoRL year or arXiv year if available)
- task_type: list, choose from {manipulation, navigation, locomotion, aerial, mobile manipulation, multi-robot, planning/control}
- problem_statement: one sentence
- method_keywords: 3–6 key terms (e.g., diffusion policy, MPC, RL, imitation, LfD, visuomotor transformer)
- robot_platforms: list (Franka, UR5, ANYmal, quadrotor, etc.)
- sensors: list (RGB, depth, LiDAR, proprioception, F/T, tactile)
- actuators: list
- environment: 'sim', 'real', or 'sim-to-real'; include simulator (Isaac, Mujoco, Habitat, Gibson) if stated
- benchmarks: array of {name, split_or_suite?, notes?} (e.g., Meta-World ML10/ML45/MT50; RLBench task suite)
- metrics: array of {name, value, higher_is_better?} (e.g., Success Rate 74%, SPL 0.62)
- baselines: list of method names compared against
- results_summary: 3–7 bullets; each bullet must include a number or metric
- limitations: 1–4 bullets
- code_url, data_url: if present
Return ONLY JSON, strictly matching the schema. If unknown, use [] or null. Avoid guesses.
"""

# ---- Robust JSON carve & normalization (same behavior as Colab) ----
def _first_balanced_json_object(text: str, max_extra_braces: int = 3):
    if "{" not in text: return None
    s = text[text.find("{"):]
    def carve(s2):
        depth, start = 0, None
        for i, ch in enumerate(s2):
            if ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return s2[start:i+1]
        return None
    cand = carve(s)
    if cand:
        try: return json.loads(cand)
        except Exception: pass
    need = s.count("{") - s.count("}")
    for k in range(1, max(need,0)+1 if need>0 else max_extra_braces+1):
        cand = carve(s + ("}" * k))
        if not cand: continue
        try: return json.loads(cand)
        except Exception: continue
    return None

def _sanitize_urls(d: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("code_url", "data_url"):
        v = d.get(k, None)
        if isinstance(v, str) and v.strip() in ("", "null", "None"):
            d[k] = None
    return d

def _coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(d, dict) and isinstance(d.get("benchmarks"), list):
        for b in d["benchmarks"]:
            if isinstance(b, dict) and isinstance(b.get("split_or_suite"), bool):
                b["split_or_suite"] = "true" if b["split_or_suite"] else "false"
    return d

class Predictor(BasePredictor):
    def setup(self):
        # Tokenizer (fast, fallback to slow)
        try:
            self.tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

        # Pick bf16 on large GPUs (A100/A10G), else 4-bit NF4
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
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else None
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb, device_map="auto"
            )
            self.mode = "4bit-NF4"

        self.pipe = pipeline("text-generation", model=mdl, tokenizer=self.tok)
        # make sure pad_token_id exists to avoid warnings
        if getattr(self.pipe.model.config, "pad_token_id", None) is None:
            self.pipe.model.config.pad_token_id = self.tok.eos_token_id

        self.kw = dict(
            max_new_tokens=640,
            do_sample=False,
            return_full_text=False,
            use_cache=False,
            eos_token_id=self.tok.eos_token_id
        )

    def predict(self,
                document: str = Input(description="Paste robotics abstract/paper text", default="")) -> Dict[str, Any]:
        # Build chat input (Qwen3 template)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document},
        ]
        text_in = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        # Generate deterministically
        with torch.inference_mode():
            out = self.pipe(text_in, **self.kw)[0]["generated_text"]

        # Carve/repair JSON, normalize, validate
        obj = _first_balanced_json_object(out)
        if obj is None:
            return {"error": "no_json_found", "tail": out[-800:]}

        obj = _sanitize_urls(_coerce_types(obj))
        try:
            parsed = Extracted.model_validate(obj)   # Pydantic v2
            return parsed.model_dump(mode="json")    # JSON-safe (HttpUrl -> str)
        except ValidationError as e:
            return {"error": "schema_validation_failed", "detail": str(e)[:500]}
