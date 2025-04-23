import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm.notebook import tqdm


def aggregate_hidden_states(hidden_states, method='max', top_n=3):
    """Aggregate hidden states array using 'max', 'mean', or 'first'."""
    stack = np.stack(hidden_states[:top_n])
    if method == 'max':
        return np.max(stack, axis=0)
    elif method == 'mean':
        return np.mean(stack, axis=0)
    elif method == 'first':
        return hidden_states[0]
    else:
        raise ValueError(f"Unknown method {method}")


def roc_auc_for_group(df, all_sources, mode="separate"):
    """Calculate ROC AUC using various per-group evaluation splits."""
    scores = {}
    if mode == "separate":
        for source in tqdm(all_sources, desc="Groupwise train/test"):
            subset = df[df.source == source]
            X = np.array([aggregate_hidden_states(hs) for hs in subset["hidden_states"]])
            y = subset["label"].apply(lambda x: int(x.lower()=="true")).values
            if len(np.unique(y)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)
                model = LogisticRegression(max_iter=10_000, fit_intercept=False, C=1e-4)
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                scores[source] = roc_auc_score(y_test, y_pred_proba)
            else:
                scores[source] = np.nan
        return scores
    elif mode == "all_vs_each":
        train_df = df.sample(frac=1.0, random_state=0).groupby(["source", "label"]).apply(
            lambda gp: gp.iloc[:int(len(gp)*0.8)]).reset_index(drop=True)
        X_train = np.array([aggregate_hidden_states(hs) for hs in train_df["hidden_states"]])
        y_train = train_df["label"].apply(lambda x: int(x.lower()=="true")).values
        model = LogisticRegression(max_iter=10_000, fit_intercept=False, C=1e-4)
        model.fit(X_train, y_train)
        test_df = df.sample(frac=1.0, random_state=0).groupby(["source", "label"]).apply(
            lambda gp: gp.iloc[int(len(gp)*0.8):]).reset_index(drop=True)
        for source in tqdm(all_sources, desc="Train-all, test-each"):
            subset = test_df[test_df["source"]==source]
            X = np.array([aggregate_hidden_states(hs) for hs in subset["hidden_states"]])
            y = subset["label"].apply(lambda x: int(x.lower()=="true")).values
            if len(np.unique(y)) > 1:
                y_pred_proba = model.predict_proba(X)[:, 1]
                scores[source] = roc_auc_score(y, y_pred_proba)
            else:
                scores[source] = np.nan
        return scores
    elif mode == "leave_one_out":
        for held_out in tqdm(all_sources, desc="Leave-one-out"):
            train = df[df.source != held_out]
            test = df[df.source == held_out]
            X_train = np.array([aggregate_hidden_states(hs) for hs in train["hidden_states"]])
            y_train = train["label"].apply(lambda x: int(x.lower()=="true")).values
            X_test = np.array([aggregate_hidden_states(hs) for hs in test["hidden_states"]])
            y_test = test["label"].apply(lambda x: int(x.lower()=="true")).values
            if (len(np.unique(y_train)) > 1) and (len(np.unique(y_test)) > 1):
                model = LogisticRegression(max_iter=10_000, fit_intercept=False, C=1e-4)
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                scores[held_out] = roc_auc_score(y_test, y_pred_proba)
            else:
                scores[held_out] = np.nan
        return scores
    else:
        raise ValueError(f"Unknown mode {mode}")


def make_bar_plot(graph1_scores, graph2_scores, graph3_scores, all_sources):
    """Plot the graphs calculated in roc_auc_for_group."""
    bar_width = 0.25
    index = np.arange(len(all_sources))

    plt.figure(figsize=(14, 8))
    g1_values = [graph1_scores.get(source, np.nan) for source in all_sources]
    g2_values = [graph2_scores.get(source, np.nan) for source in all_sources]
    g3_values = [graph3_scores.get(source, np.nan) for source in all_sources]

    plt.bar(index - bar_width, g1_values, bar_width, label="Per-Source Train/Test", color="skyblue")
    plt.bar(index, g2_values, bar_width, label="Trained on All, Test Per Source", color="orange")
    plt.bar(index + bar_width, g3_values, bar_width, label="Leave-One-Source-Out", color="green")
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random Chance (0.5)')
    plt.xlabel("Source")
    plt.ylabel("ROC AUC")
    plt.title("Testing Generalization with ROC AUC")
    plt.xticks(index, all_sources, rotation=45, ha='right')
    plt.legend(loc="lower left")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def make_heatmap(df, all_sources, aggregate_hidden_states):
    """Plot a cross dataset ROC AUC heatmap."""
    models = {}
    table = pd.DataFrame(index=all_sources, columns=all_sources, dtype=float)
    for train_source in all_sources:
        train_subset = df[df["source"] == train_source]
        X_full = np.array([aggregate_hidden_states(hs) for hs in train_subset["hidden_states"]])
        y_full = train_subset["label"].apply(lambda x: int(x.lower()=="true")).values
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
        model = LogisticRegression(
            C=1e-4, max_iter=10_000, fit_intercept=False)
        model.fit(X_train, y_train)
        models[train_source] = model
        for test_source in all_sources:
            test_subset = df[df["source"] == test_source]
            X_test = np.array([aggregate_hidden_states(hs) for hs in test_subset["hidden_states"]])
            y_test = test_subset["label"].apply(lambda x: int(x.lower()=="true")).values
            if test_source == train_source:
                X_test, y_test = X_eval, y_eval
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            table.loc[train_source, test_source] = roc_auc
    plt.figure(figsize=(10, 8))
    sns.heatmap(table, annot=True, fmt=".4f", cmap="coolwarm")
    plt.xlabel("Test Source")
    plt.ylabel("Train Source")
    plt.title("Cross-Dataset Evaluation ROC AUC")
    plt.tight_layout()
    plt.show()
    return table, models


def incremental_training_curve(df, all_sources, aggregate_hidden_states, n_runs=10):
    """Plot the ROC AUC on remaining sources as a function of the number of training sources."""
    from collections import defaultdict
    import random

    results_by_train_size = defaultdict(list)
    sources = list(all_sources)
    random.seed(0)
    for run in tqdm(range(n_runs), desc="Incremental transfer"):
        shuffled_sources = sources.copy()
        random.shuffle(shuffled_sources)
        for i in range(1, len(shuffled_sources)):
            train_sources = shuffled_sources[:i]
            test_sources = shuffled_sources[i:]
            train_subset = df[df["source"].isin(train_sources)]
            X_train = np.array([aggregate_hidden_states(hs) for hs in train_subset["hidden_states"]])
            y_train = train_subset["label"].apply(lambda x: int(x.lower()=="true")).values
            model = LogisticRegression(
                C=1e-4, max_iter=10_000, fit_intercept=False)
            model.fit(X_train, y_train)
            aucs = []
            for test_source in test_sources:
                test_subset = df[df["source"] == test_source]
                X_test = np.array([aggregate_hidden_states(hs) for hs in test_subset["hidden_states"]])
                y_test = test_subset["label"].apply(lambda x: int(x.lower()=="true")).values
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                aucs.append(roc_auc)
            avg_auc = np.mean(aucs)
            results_by_train_size[len(train_sources)].append(avg_auc)
    # Plot with error bars
    x_vals = sorted(results_by_train_size.keys())
    y_means = [np.mean(results_by_train_size[x]) for x in x_vals]
    y_stds = [np.std(results_by_train_size[x]) for x in x_vals]
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_means, marker='o', linestyle='-', color='steelblue', label="Avg ROC AUC")
    plt.fill_between(x_vals, 
                     [m - s for m, s in zip(y_means, y_stds)],
                     [m + s for m, s in zip(y_means, y_stds)],
                     color='steelblue', alpha=0.2, label="±1 std dev")
    plt.xlabel("Number of Train Sources")
    plt.ylabel("Average ROC AUC on Remaining Sources")
    plt.title(f"Incremental Training ({n_runs} Random Source Orders)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def cosine_similarity_matrix(models):
    """
    Plot cosine similarity matrix heatmap for learned coefficient vectors.
    """
    keys = list(models.keys())
    coefs = np.array([models[k].coef_[0] for k in keys])
    sim_matrix = cosine_similarity(coefs)
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", 
                cmap="coolwarm", xticklabels=keys, yticklabels=keys)
    plt.title("Cosine Similarity of Activation Vectors")
    plt.xlabel("Training Configuration")
    plt.ylabel("Training Configuration")
    plt.tight_layout()
    plt.show()


def pca_scatter_3d(X_train, y_train):
    """
    3D PCA plot of hidden activations.
    """
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_train)
    df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df['label'] = y_train
    fig = px.scatter_3d(
        df, x='PC1', y='PC2', z='PC3',
        color='label',
        title='3D PCA Scatter Plot'
    )
    fig.update_layout(width=1000, height=1000)
    fig.show()


def perform_k_means_with_labels(X_train, y_train, num_clusters):
    """
    Cluster X_train, retain indices and training labels per cluster.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_train)
    labels = kmeans.labels_
    clusters = [[] for _ in range(num_clusters)]
    cluster_labels = [[] for _ in range(num_clusters)]
    cluster_indices = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(X_train[i])
        cluster_labels[label].append(y_train[i])
        cluster_indices[label].append(i)
    return clusters, cluster_labels, kmeans, cluster_indices


def plot_clusters_3d(X_train, labels):
    """
    3D PCA visualization of clusters.
    """
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_train)
    df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df['cluster'] = labels
    fig = px.scatter_3d(
        df, x='PC1', y='PC2', z='PC3',
        color='cluster',
        color_continuous_scale='viridis',
        title='3D Cluster Scatter Plot'
    )
    fig.update_layout(width=1000, height=1000)
    fig.show()


def train_logistic_regression_per_cluster(clusters, cluster_labels):
    """
    Train a logistic regression on each cluster, returns list of models.
    """
    models = []
    for X_cluster, y_cluster in zip(clusters, cluster_labels):
        if len(X_cluster) > 0:
            model = LogisticRegression(max_iter=10_000, fit_intercept=False, C=1e-4)
            model.fit(np.stack(X_cluster), y_cluster)
            models.append(model)
        else:
            models.append(None)  # pad for clusters w/ no points
    return models


def plot_coefficient_similarity(models):
    """
    Heatmap of cosine similarity between logistic regression coefficient vectors for clusters.
    """
    # filter out None models!
    good_models = [m for m in models if m is not None]
    if not good_models:
        print("No valid models to plot similarity!")
        return
    coefs = np.array([model.coef_[0] for model in good_models])
    similarity_matrix = cosine_similarity(coefs)
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm')
    plt.title('Cosine Similarity of Logistic Regression Coefficients')
    plt.xlabel('Models')
    plt.ylabel('Models')
    plt.tight_layout()
    plt.show()


def calculate_roc_auc(models, clustering, X, y):
    """
    For each cluster, use appropriate model to get probabilities, then compute ROC AUC over all points.
    """
    cluster_labels = clustering.predict(X)
    y_pred = np.zeros_like(y, dtype=float)
    for cluster_label, model in enumerate(models):
        if model is None:
            continue
        indices = np.where(cluster_labels == cluster_label)[0]
        if len(indices) > 0:
            X_cluster_test = X[indices]
            y_prob = model.predict_proba(X_cluster_test)[:, 1]
            y_pred[indices] = y_prob
    roc_auc = roc_auc_score(y, y_pred)
    return roc_auc


def train_mixture_of_classifiers(
    model,
    X_train,
    y_train,
    epochs=500,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    scheduler_class=None,
    scheduler_kwargs=None,
    verbose=False,
):
    """
    Trains a MixtureOfClassifiers model using Torch.
    Returns the trained model.
    """
    model.train()
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (scheduler_class(optimizer, **(scheduler_kwargs or {}))) if scheduler_class else None

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            logits, _ = model(batch_X)
            loss = loss_func(logits, batch_y)
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def evaluate_by_source(model, test_df, all_sources, aggregate_hidden_states):
    """
    Evaluate the model ROC AUC on each 'source' in test_df.
    Returns: dict {source: ROC_AUC}
    """
    per_source_scores = {}
    for source in all_sources:
        df = test_df[test_df['source'] == source]
        X = torch.tensor([aggregate_hidden_states(hs) for hs in df["hidden_states"]], dtype=torch.float32)
        y = torch.tensor(df["label"].apply(lambda x: 1 if x.lower() == "true" else 0).values)
        if len(np.unique(y)) > 1:
            model.eval()
            with torch.no_grad():
                logits, _ = model(X)
                probas = torch.softmax(logits, dim=1)
                y_pred_proba = probas[:, 1].cpu().numpy()
            per_source_scores[source] = roc_auc_score(y, y_pred_proba)
        else:
            per_source_scores[source] = np.nan
    return per_source_scores


def find_best_keys_by_group(scores: dict[tuple, dict], group_index=0):
    """
    For each group (e.g. number of experts), find key with highest mean across sources.
    scores: {(num_experts, ...): {source1: auc1, ...}, ...}
    """
    grouped = defaultdict(list)
    for key, val_dict in scores.items():
        group_key = key[group_index]
        grouped[group_key].append((key, mean([v for v in val_dict.values() if not np.isnan(v)])))
    best_keys = {}
    for group_key, items in grouped.items():
        best_key, _ = max(items, key=lambda x: x[1])
        best_keys[group_key] = best_key
    return best_keys


def plot_best_key_values(scores: dict, best_keys: dict):
    """
    For each group key (num experts), plot best ROC AUC per source label.
    """
    labels = set()
    for key in best_keys.values():
        labels.update(scores[key].keys())
    labels = sorted(labels)
    x = list(best_keys.keys())
    bar_width = 0.1
    n_labels = len(labels)
    bar_positions = [i for i in range(len(x))]
    for i, label in enumerate(labels):
        heights = [scores[best_keys[group_key]].get(label, 0.0) for group_key in x]
        positions = [p + i * bar_width for p in bar_positions]
        plt.bar(positions, heights, width=bar_width, label=label)
    plt.xticks([p + bar_width * (n_labels - 1) / 2 for p in bar_positions], [str(k) for k in x])
    plt.xlabel("Number of Experts")
    plt.ylabel("ROC AUC")
    plt.title("Best Performance per Number of Experts")
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
