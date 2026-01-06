import subprocess

model_path = "YOUR MODEL PATH"
saving_path = "YOUR SAVING PATH"
model_name = "DeepSeek-R1-Distill-Qwen-7B"
data_path = "../DEmO_data"

def run_zero_main():
    cmd = [
        "python", "Zero_main.py",
        "--gpu", "0",
        "--saving_path", saving_path,
        "--data_path", data_path,
        "--task", "mmlu_math",
        "--seed", "317"
    ]
    print("=== Running Zero_main.py ===")
    subprocess.run(cmd, check=True)
    print("=== Zero_main.py finished ===\n")

def run_evaluate():
    cmd = [
        "python", "Evaluate.py",
        "--gpu", "0",
        "--model_name", model_name,
        "--saving_path", saving_path,
        "--model_path", model_path,
        "--data_path", data_path,
        "--task", "mmlu_math",
        "--method", "Zero",
        "--shots", "0",
        "--r", "1",
        "--k", "1",
        "--m", "100",
        "--max_model_len", "20000",
        "--gpu_memory_utilization", "0.83"
    ]
    print("=== Running evaluate.py ===")
    subprocess.run(cmd, check=True)
    print("=== evaluate.py finished ===\n")

if __name__ == "__main__":
    try:
        run_zero_main()
        run_evaluate()
        print("✅ All tasks completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while executing: {e.cmd}")
