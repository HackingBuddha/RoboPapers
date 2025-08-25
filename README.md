# RoboPapers (Robotics paper → structured JSON)

Extracts task type, platforms, sensors/actuators, sim/real, benchmarks, metrics, baselines, numeric results, and code/data URLs from robotics abstracts/papers.

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Decoding**: deterministic, `max_new_tokens=640`
- **JSON**: robust carving from raw text + Pydantic schema validation
