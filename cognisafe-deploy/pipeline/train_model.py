
import numpy as np
import json
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ── Output path ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_model.json")

# ── Biomarker feature order (must match anomaly.py exactly) ───────────────────
FEATURES = [
    "speech_rate", "articulation_rate", "pause_frequency",
    "pause_duration_mean", "filled_pause_rate", "pitch_mean",
    "pitch_range", "jitter", "shimmer", "HNR",
    "lexical_diversity", "semantic_coherence",
    "idea_density", "syntactic_complexity",
]

np.random.seed(42)
N = 6000  # samples per class


def make_samples(n, profile):
    """
    Generate n samples by drawing from a normal distribution
    centred on the clinical profile values with realistic variance.
    profile = { feature: (mean, std) }
    """
    data = {}
    for feat in FEATURES:
        mean, std = profile.get(feat, (0.5, 0.1))
        data[feat] = np.random.normal(mean, std, n).clip(
            0, None  # biomarkers can't be negative
        )
    return np.column_stack([data[f] for f in FEATURES])


# ── Clinical profiles per risk tier ──────────────────────────────────────────
# Values based on published cognitive speech biomarker literature.
# Each tuple: (mean, std)

GREEN_PROFILE = {
    "speech_rate":         (130, 15),
    "articulation_rate":   (5.2, 0.5),
    "pause_frequency":     (2.5, 0.8),
    "pause_duration_mean": (0.35, 0.1),
    "filled_pause_rate":   (1.2, 0.5),
    "pitch_mean":          (165, 30),
    "pitch_range":         (80, 20),
    "jitter":              (0.009, 0.003),
    "shimmer":             (0.06, 0.02),
    "HNR":                 (18.0, 2.5),
    "lexical_diversity":   (0.72, 0.08),
    "semantic_coherence":  (0.78, 0.07),
    "idea_density":        (0.44, 0.06),
    "syntactic_complexity":(4.2, 0.8),
}

YELLOW_PROFILE = {
    "speech_rate":         (108, 18),   # slower
    "articulation_rate":   (4.5, 0.6),
    "pause_frequency":     (4.2, 1.0),  # more pauses
    "pause_duration_mean": (0.55, 0.15),
    "filled_pause_rate":   (2.5, 0.8),
    "pitch_mean":          (155, 35),
    "pitch_range":         (60, 20),
    "jitter":              (0.016, 0.005),
    "shimmer":             (0.10, 0.03),
    "HNR":                 (14.0, 2.5),
    "lexical_diversity":   (0.58, 0.09),
    "semantic_coherence":  (0.62, 0.09),
    "idea_density":        (0.34, 0.07),
    "syntactic_complexity":(3.2, 0.9),
}

ORANGE_PROFILE = {
    "speech_rate":         (88, 20),    # noticeably slower
    "articulation_rate":   (3.8, 0.7),
    "pause_frequency":     (6.5, 1.5),
    "pause_duration_mean": (0.80, 0.2),
    "filled_pause_rate":   (4.0, 1.2),
    "pitch_mean":          (145, 35),
    "pitch_range":         (45, 18),
    "jitter":              (0.024, 0.007),
    "shimmer":             (0.15, 0.04),
    "HNR":                 (10.5, 3.0),
    "lexical_diversity":   (0.46, 0.10),
    "semantic_coherence":  (0.48, 0.10),
    "idea_density":        (0.26, 0.07),
    "syntactic_complexity":(2.5, 1.0),
}

RED_PROFILE = {
    "speech_rate":         (68, 22),    # severely slow
    "articulation_rate":   (3.0, 0.8),
    "pause_frequency":     (9.5, 2.0),
    "pause_duration_mean": (1.2, 0.3),
    "filled_pause_rate":   (6.5, 1.8),
    "pitch_mean":          (130, 35),
    "pitch_range":         (30, 15),
    "jitter":              (0.034, 0.009),
    "shimmer":             (0.22, 0.05),
    "HNR":                 (7.0, 3.0),
    "lexical_diversity":   (0.34, 0.10),
    "semantic_coherence":  (0.34, 0.10),
    "idea_density":        (0.18, 0.06),
    "syntactic_complexity":(1.8, 0.8),
}


def main():
    print("[Train] Generating synthetic population data...")

    X_green  = make_samples(N,          GREEN_PROFILE)
    X_yellow = make_samples(N,          YELLOW_PROFILE)
    X_orange = make_samples(int(N*0.6), ORANGE_PROFILE)  # less common
    X_red    = make_samples(int(N*0.3), RED_PROFILE)     # rare

    X = np.vstack([X_green, X_yellow, X_orange, X_red])
    y_raw = (
        ["Green"]  * N +
        ["Yellow"] * N +
        ["Orange"] * int(N * 0.6) +
        ["Red"]    * int(N * 0.3)
    )

    # Encode labels: Green=0, Orange=1, Red=2, Yellow=3
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    print(f"[Train] Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[Train] Class distribution: {dict(zip(*np.unique(y_raw, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[Train] Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print("\n[Train] Evaluation on test set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the model
    model.save_model(MODEL_PATH)
    print(f"\n[Train] ✅ Model saved to {MODEL_PATH}")

    # Save label mapping so anomaly.py can decode predictions
    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    map_path  = MODEL_PATH.replace(".json", "_labels.json")
    with open(map_path, "w") as f:
        json.dump(label_map, f)
    print(f"[Train] ✅ Label map saved to {map_path}")

    # Print feature importances
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    print("\n[Train] Feature importances:")
    for feat, imp in ranked:
        bar = "█" * int(imp * 100)
        print(f"  {feat:<25} {imp:.4f}  {bar}")


if __name__ == "__main__":
    main()