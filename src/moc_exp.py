# moc_exp.py
"""
Train and evaluate a mixture of classifiers.
"""


import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
from tqdm import trange
from utils import (
    aggregate_hidden_states,
    train_mixture_of_classifiers,
    evaluate_by_source,
    find_best_keys_by_group,
    plot_best_key_values,
)
from moc_model import MixtureOfClassifiers


# Prepare data
SAMPLE_SIZE = 500
cls_df = pd.read_parquet("data/truth_train_cls_dataset.parquet")
cls_df["source"] = cls_df.source.apply(lambda x: x.split("_")[0] if (("train" in x) or ("dev" in x) or ("test" in x)) else x).reset_index(drop=True)
cls_df = (
    cls_df.drop_duplicates(subset=["Question", "Answer"])
    .groupby(["source", "label"])
    .apply(lambda gp: gp.sample(n=SAMPLE_SIZE, random_state=0) if len(gp) > SAMPLE_SIZE else gp)
    .reset_index(drop=True)
)
train_df = cls_df.sample(frac=1.0, random_state=0).groupby(["source", "label"]).apply(lambda gp: gp.iloc[:int(len(gp) * 0.8)]).reset_index(drop=True)
test_df  = cls_df.sample(frac=1.0, random_state=0).groupby(["source", "label"]).apply(lambda gp: gp.iloc[int(len(gp) * 0.8):]).reset_index(drop=True)
X_train = torch.tensor([aggregate_hidden_states(hs) for hs in train_df["hidden_states"]], dtype=torch.float32)
y_train = torch.tensor(train_df["label"].apply(lambda x: 1 if x.lower() == "true" else 0).values, dtype=torch.long)

input_dim = X_train.shape[1]
all_sources = sorted(cls_df.source.unique())
all_num_experts = [1, 2, 3, 5, 7]
weight_decays = [1e-4]
lrs = [1e-3]
epochs = 500

all_graph_scores = {}
all_models = {}

run_num = 1
num_rounds = len(all_num_experts) * len(weight_decays) * len(lrs)

for num_experts in all_num_experts:
    for weight_decay in weight_decays:
        for lr in lrs:
            print(f"Run {run_num} / {num_rounds}: experts={num_experts} lr={lr} wd={weight_decay}")
            
            # Create new model
            model = MixtureOfClassifiers(input_dim, num_experts)
            # Train model
            model = train_mixture_of_classifiers(
                model=model,
                X_train=X_train,
                y_train=y_train,
                epochs=epochs,
                batch_size=len(X_train),
                lr=lr,
                weight_decay=weight_decay,
                scheduler_class=CosineAnnealingLR,
                scheduler_kwargs=dict(T_max=epochs, eta_min=lr / 100),
                verbose=False
            )
            # Evaluate
            graph_scores = evaluate_by_source(model, test_df, all_sources, aggregate_hidden_states)
            all_graph_scores[(num_experts, weight_decay, lr)] = graph_scores
            all_models[(num_experts, weight_decay, lr)] = model
            run_num += 1

# Save results
with open("all_graph_scores.pkl", "wb") as f:
    pickle.dump(all_graph_scores, f)
with open("all_models_gumbel.pkl", "wb") as f:
    pickle.dump(all_models, f)

# Load and analyze
with open("all_graph_scores.pkl", "rb") as f:
    all_graph_scores = pickle.load(f)

best = find_best_keys_by_group(all_graph_scores)
plot_best_key_values(all_graph_scores, best)
