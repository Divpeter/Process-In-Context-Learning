# Process In-context Learning

This code contains the implementation of Process In-Context Learning (PICL), a novel approach for enhancing language model performance on complex reasoning tasks through dynamic demonstration selection and insertion.

## Requirements

Before running the code, please install the following dependencies: `vllm` , `FlagEmbedding`, `transformers`

## Running the Code

We provide running scripts for three methods: **Zero-shot**, **Similarity-few-shot**, and **PICL**. Ensure the paths provided to all parameters are correct for your system

```python
cd PICL
python run_Zero.py
python run_Similarity.py
python run_Picl.py
```

### Parameters Explanation

After retrieving the demonstrations, run the main PICL method using `picl_main.py`:

```
--gpu <gpu_id> \
--model_name <model_name> \
--saving_path <demonstrations_directory> \
--model_path <model_path> \
--data_path <dataset_path> \
--task <dataset_name> \
--method <method_name>\
--shots <few_shots_number>\
--r <max_interventions> \
--k <insertion_count> \
--m <dataset_size> \
--max_model_len <maximum_sequence_length> \
--gpu_memory_utilization <gpu_usage_ratio>
```

## Notes

- Ensure the paths provided to all parameters are correct for your system
-  `--r` and `--k` are two core parameters used for PICL