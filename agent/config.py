from dataclasses import dataclass

@dataclass
class Configs:
    span_extraction_model = "llama-3.3-70b-versatile"
    reasoning_model = "qwen/qwen3-32b"
    pubmed_model_dir = "outputs/tier1_pubmedbert/model/"
    tier5_agent_model = "qwen/qwen3-32b"
    gpt_120b_oss_model = "openai/gpt-oss-120b"
    