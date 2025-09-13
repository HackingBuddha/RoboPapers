from typing import Optional, List, Dict, Any
from cog import BasePredictor, Input
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import json
import time

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Pydantic Models
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

# Improved System Prompt with Examples
SYSTEM_PROMPT = """You are a robotics paper metadata extractor. Extract information from robotics papers and return ONLY a valid JSON object. No additional text.

Required JSON format:
{
  "title": "Paper title",
  "venue_year": "ICRA 2024" or "arXiv 2024" or null,
  "task_type": ["manipulation", "navigation", "locomotion", "aerial", "mobile manipulation", "multi-robot", "planning/control"],
  "problem_statement": "One sentence describing the problem",
  "method_keywords": ["keyword1", "keyword2"],
  "robot_platforms": ["Franka", "UR5", "ANYmal"],
  "sensors": ["RGB", "depth", "LiDAR"],
  "actuators": ["7-DoF arm"],
  "environment": "sim", "real", or "sim-to-real",
  "benchmarks": [{"name": "Meta-World", "notes": "ML10 tasks"}],
  "metrics": [{"name": "success rate", "value": "85%", "higher_is_better": true}],
  "baselines": ["method1", "method2"],
  "results_summary": ["Result with numbers"],
  "limitations": ["limitation1"],
  "code_url": null,
  "data_url": null
}

EXAMPLE:
Input: "We present DiffusionPolicy for robotic manipulation using a 7-DoF Franka arm with RGB-D cameras. Our method achieves 94% success on Meta-World ML10 tasks, outperforming Behavior Cloning by 15%. Limitations include poor performance in low-light conditions."

Output: {"title": "DiffusionPolicy for Robotic Manipulation", "venue_year": null, "task_type": ["manipulation"], "problem_statement": "We present DiffusionPolicy for robotic manipulation using diffusion models.", "method_keywords": ["diffusion policy", "imitation learning"], "robot_platforms": ["Franka"], "sensors": ["RGB", "depth"], "actuators": ["7-DoF arm"], "environment": "sim", "benchmarks": [{"name": "Meta-World ML10"}], "metrics": [{"name": "success rate", "value": "94%", "higher_is_better": true}], "baselines": ["Behavior Cloning"], "results_summary": ["Achieves 94% success on Meta-World ML10 tasks", "Outperforms Behavior Cloning by 15%"], "limitations": ["Poor performance in low-light conditions"], "code_url": null, "data_url": null}

Return ONLY the JSON object. No explanation."""

def extract_and_repair_json(text: str) -> Optional[Dict]:
    """Advanced JSON extraction with multiple repair strategies"""
    if "{" not in text:
        return None
        
    start_idx = text.find("{")
    text_from_brace = text[start_idx:]
    
    # Strategy 1: Try parsing as-is first
    try:
        return json.loads(text_from_brace)
    except:
        pass
    
    # Strategy 2: Balance braces
    depth = 0
    end_idx = None
    for i, char in enumerate(text_from_brace):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    
    if end_idx:
        candidate = text_from_brace[:end_idx]
        try:
            return json.loads(candidate)
        except:
            pass
    
    # Strategy 3: Add missing braces
    brace_count = text_from_brace.count("{") - text_from_brace.count("}")
    if brace_count > 0:
        candidate = text_from_brace + "}" * brace_count
        try:
            return json.loads(candidate)
        except:
            pass
    
    # Strategy 4: Try finding last valid closing brace
    for i in range(len(text_from_brace) - 1, 0, -1):
        if text_from_brace[i] == "}":
            candidate = text_from_brace[:i+1]
            try:
                return json.loads(candidate)
            except:
                continue
    
    return None

def sanitize_urls(data: Dict) -> Dict:
    """Convert empty-ish URLs to None for Pydantic validation"""
    url_fields = ["code_url", "data_url"]
    for field in url_fields:
        if field in data and isinstance(data[field], str):
            val = data[field].strip().lower()
            if val in ["", "null", "none", "n/a", "na", "not available", "not provided"]:
                data[field] = None
        elif field in data and not data[field]:
            data[field] = None
    return data

def coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
    """Fix type mismatches for Pydantic validation"""
    if isinstance(d, dict) and isinstance(d.get("benchmarks"), list):
        for b in d["benchmarks"]:
            if isinstance(b, dict) and isinstance(b.get("split_or_suite"), bool):
                b["split_or_suite"] = "true" if b["split_or_suite"] else "false"
    return d

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and tokenizer - optimized for caching"""
        import os
        
        # Set cache directory to persistent storage
        cache_dir = "/src/.cache/huggingface"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        
        print(f"Loading model {MODEL_ID}...")
        
        # Load tokenizer with retry logic
        for attempt in range(3):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_ID, 
                    use_fast=True,
                    cache_dir=cache_dir
                )
                break
            except Exception as e:
                print(f"Tokenizer attempt {attempt+1} failed: {e}")
                if attempt == 2:  # Last attempt, try slow tokenizer
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            MODEL_ID, 
                            use_fast=False,
                            cache_dir=cache_dir
                        )
                    except Exception as e2:
                        raise Exception(f"Failed to load tokenizer: {e2}")

        # Detect GPU capabilities
        gpu_props = None
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            gpu_props = (device_props.name, round(device_props.total_memory/(1024**3), 2))
            print(f"GPU detected: {gpu_props[0]} with {gpu_props[1]}GB memory")

        # Load model with appropriate precision
        model_start = time.time()
        if gpu_props and ("A100" in gpu_props[0] or "H100" in gpu_props[0] or gpu_props[1] >= 40):
            print("Using bf16 + TF32 optimization for high-end GPU")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.bfloat16, 
                device_map="cuda",
                cache_dir=cache_dir
            )
            self.mode = "bf16+TF32"
        else:
            print("Using 4-bit quantization for standard GPU")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, 
                quantization_config=bnb_config, 
                device_map="auto",
                cache_dir=cache_dir
            )
            self.mode = "4bit-NF4"
        
        model_load_time = time.time() - model_start
        print(f"Model loaded in {model_load_time:.1f}s")

        # Create pipeline
        self.pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        # Set pad token if missing
        if getattr(self.pipeline.model.config, "pad_token_id", None) is None:
            self.pipeline.model.config.pad_token_id = self.tokenizer.eos_token_id
            if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Generation settings
        self.generation_kwargs = {
            "max_new_tokens": 1000,
            "do_sample": False,
            "return_full_text": False,
            "use_cache": True,  # Enable caching for faster inference
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "temperature": None,  # Avoid temperature warning in greedy mode
            "top_p": None,       # Avoid top_p warning in greedy mode
        }
        
        print(f"Setup complete in {self.mode} mode. Ready for inference.")
        
        # Warm up with a test inference to cache everything
        try:
            print("Warming up model...")
            test_msgs = [{"role": "system", "content": "Test"}, {"role": "user", "content": "Hello"}]
            test_input = self.tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
            with torch.inference_mode():
                _ = self.pipeline(test_input, max_new_tokens=10, do_sample=False, return_full_text=False)
            print("Warmup complete.")
        except Exception as e:
            print(f"Warmup failed but continuing: {e}")

    def predict(
        self,
        document: str = Input(
            description="Paste robotics paper text (abstract, full paper, or key sections)",
            default=""
        )
    ) -> Dict[str, Any]:
        """Extract structured metadata from robotics paper text"""
        
        if not document.strip():
            return {"error": "empty_input", "message": "Please provide paper text to analyze"}

        # Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document.strip()}
        ]
        
        # Apply chat template
        try:
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            return {"error": "template_failed", "message": str(e)}

        # Generate response
        start_time = time.time()
        try:
            with torch.inference_mode():
                output = self.pipeline(text_input, **self.generation_kwargs)[0]["generated_text"]
        except Exception as e:
            return {"error": "generation_failed", "message": str(e)}
        
        generation_time = round(time.time() - start_time, 2)

        # Extract JSON from output
        parsed_json = extract_and_repair_json(output)
        if parsed_json is None:
            return {
                "error": "json_extraction_failed",
                "raw_output": output[:1000],
                "generation_time": generation_time,
                "mode": self.mode
            }

        # Sanitize and validate
        try:
            sanitized = sanitize_urls(coerce_types(parsed_json))
            validated = Extracted.model_validate(sanitized)
            result = validated.model_dump(mode="json")
            
            # Add metadata
            result["_metadata"] = {
                "generation_time": generation_time,
                "mode": self.mode,
                "success": True
            }
            
            return result
            
        except ValidationError as e:
            return {
                "error": "validation_failed",
                "raw_json": parsed_json,
                "validation_errors": str(e)[:1000],
                "generation_time": generation_time,
                "mode": self.mode
            }
        except Exception as e:
            return {
                "error": "processing_failed", 
                "message": str(e),
                "generation_time": generation_time,
                "mode": self.mode
            }
# --- Pydantic v1/v2 compatibility helpers ---
def _pd_v2():
    import pydantic as _pd
    try:
        return int(_pd.__version__.split(".")[0]) >= 2
    except Exception:
        return False

# later, where you currently do:
# # --- begin compat helpers ---
def _pydantic_parse(model_cls, data):
    """Works on Pydantic v1 and v2."""
    # v2: BaseModel.model_validate
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    # v1: BaseModel.parse_obj
    return model_cls.parse_obj(data)

def _pydantic_to_dict(model_instance):
    """Works on Pydantic v1 and v2."""
    if hasattr(model_instance, "model_dump"):   # v2
        return model_instance.model_dump()
    return model_instance.dict()                # v1
# --- end compat helpers ---

# use it where you validate/serialize:
parsed = _pydantic_parse(Extracted, obj)
return _pydantic_to_dict(parsed)

# return parsed.model_dump(mode="json")

if _pd_v2():
    parsed = Extracted.model_validate(obj)        # v2
    out = parsed.model_dump(mode="json")          # v2
else:
    parsed = Extracted.parse_obj(obj)             # v1
    out = parsed.dict()                           # v1

return out

