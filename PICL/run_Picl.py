import subprocess

saving_path = "YOUR SAVING PATH"
embedding_model_path = "YOUR EMBEDDING MODEL PATH"
model_name = "DeepSeek-R1-Distill-Qwen-7B"
model_path = "YOUR MODEL PATH"
data_path = "../DEmO_data"

def run_semantic_retrieval():
    cmd = [
        "python", "Semantic_retrieval.py",
        "--gpu", "0",
        "--saving_path", saving_path,
        "--data_path", data_path,
        "--task", "mmlu_math",
        "--shots", "32",
        "--embedding_model_path", embedding_model_path,
        "--type", "reason"
    ]
    print("=== Running Semantic_retrieval.py ===")
    subprocess.run(cmd, check=True)
    print("=== Semantic_retrieval.py finished ===\n")


def run_evaluate():
    cmd = [
        "python", "Evaluate.py",
        "--gpu", "0",
        "--model_name", model_name,
        "--saving_path", saving_path,
        "--model_path", model_path,
        "--data_path", data_path,
        "--task", "mmlu_math",
        "--method", "Picl",
        "--shots", "4",
        "--r", "1",
        "--k", "1",
        "--m", "100",
        "--max_model_len", "20000",
        "--gpu_memory_utilization", "0.83"
    ]
    print("=== Running Evaluate.py ===")
    subprocess.run(cmd, check=True)
    print("=== Evaluate.py finished ===\n")


if __name__ == "__main__":
    try:
        run_semantic_retrieval()
        run_evaluate()
        print("✅ All tasks completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while executing: {e.cmd}")
