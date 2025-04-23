# annotation.py
"""
Annotates LLM-generated outputs for factual consistency with LM Studio or other LLM APIs.

Outputs:
- Parquet file with True/False labels for each generation.
"""

import pandas as pd
from tqdm import tqdm
import requests

def remove_duplicates(row: dict, cols_to_check: list[str]) -> dict:
    seen = set()
    for col in cols_to_check:
        val = row[col].strip() if isinstance(row[col], str) else row[col]
        if val in seen:
            row[col] = None
        else:
            seen.add(val)
    return row

def prepare_prompts(df: pd.DataFrame, num_generations: int) -> tuple[list[str], list[tuple[int, str, str]]]:
    prompts, mapping = [], []
    for idx, row in df.iterrows():
        for i in range(1, num_generations + 1):
            gen_col = f"Generation_{i}"
            label_col = f"label_{i}"
            if pd.notna(row[gen_col]):
                prompt = f"""
Ground Truth Answer: {row['Answer'].strip().lower()}
Generated Answer: {row[gen_col].strip().lower()}

Does the generated answer convey the same meaning as the ground truth answer? 
- Ignore prefixes, suffixes, formatting, and minor variations. 
- Consider only the core meaning, not the exact wording. 
- Respond strictly with 'True' or 'False'.
"""
                prompts.append(prompt)
                mapping.append((idx, label_col, row[gen_col]))
    return prompts, mapping

def lmstudio_llm_call(prompts: list[str], model_name: str = "gemma-3-12b-it", port: int = 1234) -> list[str | None]:
    # Requires LM Studio running API with given model.
    API_URL = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    responses = []
    payloads = [
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        for prompt in prompts
    ]

    for payload in tqdm(payloads, desc="Annotating"):
        response = requests.post(API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            responses.append(
                response.json()["choices"][0]["message"]["content"].strip()
            )
        else:
            print("Error:", response.text)
            responses.append(None)
    return responses

def main():
    source = "all"
    num_generations = 20
    parquet_input = f"data/truth_{source}_with_generations.parquet"
    parquet_output = f"data/truth_{source}_with_generations_labels.parquet"

    df = pd.read_parquet(parquet_input)
    cols_to_check = [f"Generation_{i}" for i in range(1, num_generations + 1)]
    label_cols = [f"label_{i}" for i in range(1, num_generations + 1)]
    for col in label_cols:
        df[col] = None
    
    # Remove row duplicates across generations for each sample
    df[cols_to_check] = df[cols_to_check].apply(lambda row: remove_duplicates(row, cols_to_check), axis=1)
    print("To annotate:", df[cols_to_check].count().sum())

    prompts, mapping = prepare_prompts(df, num_generations)
    responses = lmstudio_llm_call(prompts)

    for (idx, label_col, _), response in zip(mapping, responses):
        df.at[idx, label_col] = response

    df.to_parquet(parquet_output)
    print("Annotation complete! Output at:", parquet_output)

if __name__ == "__main__":
    main()
