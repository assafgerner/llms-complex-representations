# generation.py
"""
Generates model outputs and hidden states for a dataset of questions.

Outputs:
- Parquet files with multiple generations per question and their hidden states.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def sample_by_group(df: pd.DataFrame, group_col: list[str] | str, n: int, random_state: int = 0) -> pd.DataFrame:
    return df.groupby(group_col, group_keys=False).apply(
        lambda gp: gp.sample(n, random_state=random_state) if len(gp) > n else gp
    ).reset_index(drop=True)

def generate_and_save_hidden_states_batch(
    questions: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    layer: int = 20,
    max_new_tokens: int = 16,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    batch_size: int= 8
) -> None:
    # Prepare prompts for batch
    input_texts = [
        q + ". Provide a single concise answer and write nothing else.\nAnswer: **"
        for q in questions
    ]

    input_ids = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    outputs = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    # Hidden states: last token of the specified layer, per sample.
    hidden_states_batch = [
        np.stack([
            h_s[layer][i, -1, :].squeeze().cpu().numpy()
            for h_s in outputs.hidden_states
        ])
        for i in range(len(questions))
    ]

    # Get the generated output text after the prompt
    answers_batch = [
        tokenizer.decode(seq.squeeze(), skip_special_tokens=True).split(it)[1]
        for seq, it in zip(outputs.sequences, input_texts)
    ]

    return answers_batch, hidden_states_batch

def main():
    source = "all"
    input_data_path = f"data/truth_{source}.parquet"
    output_prefix = f"data/truth_{source}_with_generations"
    num_samples = 3000
    num_generations = 20
    batch_size = 16
    device = "mps"

    # Load data
    df = pd.read_parquet(input_data_path)

    # Stratified sampling
    df_sampled = sample_by_group(df, "source", num_samples)
    print(df_sampled.shape)
    print(df["source"].value_counts())

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.float16,
        device_map=device,
    )

    for gen_num in range(1, num_generations + 1):
        answers_agg, hidden_states_agg = [], []

        question_batches = [
            df_sampled["Question"].tolist()[i:i + batch_size]
            for i in range(0, len(df_sampled), batch_size)
        ]

        for question_batch in tqdm(question_batches, desc=f"Generation {gen_num}"):
            answers_batch, hidden_states_batch = generate_and_save_hidden_states_batch(
                question_batch, model, tokenizer, device, batch_size=batch_size, max_new_tokens=8
            )
            answers_agg.extend(answers_batch)
            hidden_states_agg.extend(hidden_states_batch)

        df_sampled[f"Generation_{gen_num}"] = answers_agg
        df_sampled[f"hidden_states_{gen_num}"] = hidden_states_agg

        # Save after each generation
        temp_df = df_sampled[['Question', 'Answer', 'source', f'Generation_{gen_num}', f'hidden_states_{gen_num}']]
        temp_df[f'hidden_states_{gen_num}'] = temp_df[f'hidden_states_{gen_num}'].apply(lambda x: x if isinstance(x, list) else x.tolist())
        temp_df.to_parquet(f"{output_prefix}_{gen_num}.parquet")

    # Optionally, stitch generations together (single file with all outputs)
    df_all = pd.read_parquet(f"{output_prefix}_1.parquet").drop(columns=["hidden_states_1"])
    for i in range(2, num_generations + 1):
        temp_df = pd.read_parquet(f"{output_prefix}_{i}.parquet")[[f"Generation_{i}"]]
        df_all = pd.concat([df_all, temp_df], axis=1)
    df_all.to_parquet(f"{output_prefix}.parquet")
    print("Generation complete!")

if __name__ == "__main__":
    main()
