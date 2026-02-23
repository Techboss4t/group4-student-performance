"""
predictor.py v5.0
Train: uses scores + lifestyle + actual result from past students
Predict: takes ONLY lifestyle factors → predicts Pass/Fail + CGPA + Risk
"""

import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

INCOME_MAP = {"low": 0, "middle": 1, "high": 2}

# Features used for PREDICTION (lifestyle only — no scores needed)
PREDICT_FEATURES = [
    "study_hours",
    "family_income_enc",
    "has_part_time_job",
    "mental_health",
    "has_internet",
    "carryover_subjects",
]

# Features used for TRAINING (includes scores so AI learns the full picture)
TRAIN_FEATURES = [
    "calc_score", "physics_score", "chem_score", "prog_score", "stat_score",
    "attendance", "study_hours", "family_income_enc",
    "has_part_time_job", "mental_health", "has_internet", "carryover_subjects",
]


# ── Risk Engine ───────────────────────────────────────────────
def compute_risk(data: dict) -> tuple:
    score = 0
    recs  = []

    hrs     = float(data.get("study_hours", 8))
    income  = data.get("family_income", "middle")
    partjob = int(data.get("has_part_time_job", 0))
    mental  = int(data.get("mental_health", 5))
    inet    = int(data.get("has_internet", 1))
    carry   = int(data.get("carryover_subjects", 0))

    # Study hours (max 30)
    if hrs < 3:
        score += 30
    elif hrs < 6:
        score += 18
    elif hrs < 10:
        score += 8

    # Carry-over subjects (max 25)
    if carry >= 4:
        score += 25
    elif carry >= 2:
        score += 15
    elif carry >= 1:
        score += 8

    # Mental health (max 20)
    if mental <= 2:
        score += 20
    elif mental <= 4:
        score += 12
    elif mental <= 6:
        score += 5

    # Family income (max 10)
    if income == "low":
        score += 10
    elif income == "middle":
        score += 3

    # Part-time job (max 10)
    if partjob:
        score += 10

    # No internet (max 5)
    if not inet:
        score += 5

    score = min(score, 100)

    if score >= 55:   level = "HIGH"
    elif score >= 25: level = "MEDIUM"
    else:             level = "LOW"

    return level, score


def cgpa_from_lifestyle(data: dict, pass_prob: float) -> float:
    """
    Estimate predicted CGPA from lifestyle factors + pass probability.
    Higher pass_prob → higher predicted CGPA.
    """
    hrs    = float(data.get("study_hours", 8))
    income = data.get("family_income", "middle")
    mental = int(data.get("mental_health", 5))
    inet   = int(data.get("has_internet", 1))
    partjob= int(data.get("has_part_time_job", 0))
    carry  = int(data.get("carryover_subjects", 0))

    # Base from pass probability
    base = 1.0 + (pass_prob / 100) * 4.0   # 0→1.0, 100→5.0

    # Lifestyle adjustments
    adj  = (hrs - 8) * 0.05                  # more study = higher CGPA
    adj += (mental - 5) * 0.06               # better mental = higher
    adj += (INCOME_MAP.get(income, 1) - 1) * 0.08
    adj += inet * 0.05
    adj -= partjob * 0.12
    adj -= carry * 0.10

    return round(min(5.0, max(0.5, base + adj)), 2)


# ── ML Model ──────────────────────────────────────────────────
class StudentPredictor:
    def __init__(self):
        self.rf  = None
        self.gb  = None
        self.lr  = None
        self.scaler     = None
        self.is_trained = False
        self.trained_on = "none"
        self.sample_count = 0
        self.rf_acc = 0
        self.gb_acc = 0
        self.lr_acc = 0
        # Train on synthetic data immediately so predictions work from day 1
        self._train_synthetic()

    def _row_to_train_features(self, r: dict) -> list:
        return [
            float(r.get("calc_score", 0)),
            float(r.get("physics_score", 0)),
            float(r.get("chem_score", 0)),
            float(r.get("prog_score", 0)),
            float(r.get("stat_score", 0)),
            float(r.get("attendance", 75)),
            float(r.get("study_hours", 8)),
            float(INCOME_MAP.get(r.get("family_income","middle"), 1)),
            float(r.get("has_part_time_job", 0)),
            float(r.get("mental_health", 5)),
            float(r.get("has_internet", 1)),
            float(r.get("carryover_subjects", 0)),
        ]

    def _row_to_predict_features(self, r: dict) -> list:
        return [
            float(r.get("study_hours", 8)),
            float(INCOME_MAP.get(r.get("family_income","middle"), 1)),
            float(r.get("has_part_time_job", 0)),
            float(r.get("mental_health", 5)),
            float(r.get("has_internet", 1)),
            float(r.get("carryover_subjects", 0)),
        ]

    def _generate_synthetic(self, n=1500):
        """
        Generate synthetic training data.
        Lifestyle factors correlate with Pass/Fail so the model can learn
        to predict from lifestyle alone.
        """
        np.random.seed(42)
        rows, labels = [], []
        for _ in range(n):
            hrs     = np.random.uniform(1, 20)
            income  = np.random.choice(["low","middle","high"], p=[0.3,0.5,0.2])
            mental  = int(np.clip(np.random.normal(6,2), 1, 10))
            partjob = int(np.random.random() < 0.3)
            inet    = int(np.random.random() < 0.75)
            carry   = int(np.random.choice([0,1,2,3,4], p=[0.5,0.25,0.14,0.07,0.04]))
            att     = float(np.clip(np.random.normal(70, 15), 40, 100))

            # Lifestyle bonus affects scores
            bonus = ((hrs-8)*1.5 + (mental-5)*1.2
                     + (INCOME_MAP[income]-1)*1.5
                     - partjob*3.0 + (inet-0.5)*1.0
                     - carry*2.0)

            scores = {
                "calc_score":     float(np.clip(np.random.normal(58+bonus, 15), 10, 100)),
                "physics_score":  float(np.clip(np.random.normal(58+bonus, 15), 10, 100)),
                "chem_score":     float(np.clip(np.random.normal(58+bonus, 15), 10, 100)),
                "prog_score":     float(np.clip(np.random.normal(58+bonus, 15), 10, 100)),
                "stat_score":     float(np.clip(np.random.normal(58+bonus, 15), 10, 100)),
            }
            passed = all(v >= 40 for v in scores.values())

            rows.append({**scores,
                "attendance": att, "study_hours": hrs, "family_income": income,
                "has_part_time_job": partjob, "mental_health": mental,
                "has_internet": inet, "carryover_subjects": carry,
            })
            labels.append(1 if passed else 0)
        return rows, labels

    def _fit(self, rows, labels, feature_fn):
        X = np.array([feature_fn(r) for r in rows])
        y = np.array(labels)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        self.scaler = StandardScaler()
        Xtr_sc = self.scaler.fit_transform(X_tr)
        Xte_sc = self.scaler.transform(X_te)

        self.rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.lr = LogisticRegression(max_iter=1000, random_state=42)

        self.rf.fit(X_tr, y_tr)
        self.gb.fit(X_tr, y_tr)
        self.lr.fit(Xtr_sc, y_tr)

        self.rf_acc = round(accuracy_score(y_te, self.rf.predict(X_te))*100, 2)
        self.gb_acc = round(accuracy_score(y_te, self.gb.predict(X_te))*100, 2)
        self.lr_acc = round(accuracy_score(y_te, self.lr.predict(Xte_sc))*100, 2)
        self.sample_count = len(rows)
        self.is_trained = True

    def _train_synthetic(self):
        rows, labels = self._generate_synthetic()
        self._fit(rows, labels, self._row_to_train_features)
        self.trained_on = "synthetic"
        print(f"  ✔  AI trained on synthetic data — RF:{self.rf_acc}% GB:{self.gb_acc}% LR:{self.lr_acc}%")

    def train_on_real_data(self, db_rows: list) -> bool:
        """Train on real past student data from DB."""
        if len(db_rows) < 20:
            print(f"  ⚠  Need 20+ students to retrain. Have {len(db_rows)}.")
            return False
        labels = [1 if r.get("actual_result","FAIL")=="PASS" else 0 for r in db_rows]
        self._fit(db_rows, labels, self._row_to_train_features)
        self.trained_on = f"real ({len(db_rows)} students)"
        print(f"  ✔  AI retrained on REAL DATA (n={len(db_rows)}) — RF:{self.rf_acc}% GB:{self.gb_acc}% LR:{self.lr_acc}%")
        return True

    def predict(self, data: dict) -> dict:
        """
        Predict from lifestyle factors only.
        Uses PREDICT_FEATURES subset — no scores needed.
        """
        # Build feature vector using predict features only
        feats = np.array([self._row_to_predict_features(data)]).reshape(1, -1)

        # We need to match the scaler's expected features (trained on full features)
        # Pad with median values for the score/attendance columns
        full_feats = np.array([[
            58.0, 58.0, 58.0, 58.0, 58.0,  # scores set to median
            72.0,                            # attendance median
            float(data.get("study_hours", 8)),
            float(INCOME_MAP.get(data.get("family_income","middle"), 1)),
            float(data.get("has_part_time_job", 0)),
            float(data.get("mental_health", 5)),
            float(data.get("has_internet", 1)),
            float(data.get("carryover_subjects", 0)),
        ]])

        full_sc = self.scaler.transform(full_feats)

        rf_p  = self.rf.predict_proba(full_feats)[0]
        gb_p  = self.gb.predict_proba(full_feats)[0]
        lr_p  = self.lr.predict_proba(full_sc)[0]

        # Ensemble
        ens_pass = round((0.4*rf_p[1] + 0.4*gb_p[1] + 0.2*lr_p[1])*100, 1)
        ens_fail = round(100 - ens_pass, 1)
        pred     = "PASS" if ens_pass >= 50 else "FAIL"

        # Confidence
        if ens_pass >= 80 or ens_pass <= 20: conf = "HIGH"
        elif ens_pass >= 65 or ens_pass <= 35: conf = "MEDIUM"
        else: conf = "LOW"

        pred_cgpa = cgpa_from_lifestyle(data, ens_pass)
        risk_level, risk_score = compute_risk(data)

        return {
            "predicted_result":  pred,
            "predicted_cgpa":    pred_cgpa,
            "pass_probability":  ens_pass,
            "fail_probability":  ens_fail,
            "risk_level":        risk_level,
            "risk_score":        risk_score,
            "model_confidence":  conf,
            "rf_pass":           round(rf_p[1]*100, 1),
            "gb_pass":           round(gb_p[1]*100, 1),
            "lr_pass":           round(lr_p[1]*100, 1),
            "trained_on":        self.trained_on,
            "rf_acc":            self.rf_acc,
            "gb_acc":            self.gb_acc,
            "lr_acc":            self.lr_acc,
        }

    def status_info(self):
        return {
            "is_trained":   self.is_trained,
            "trained_on":   self.trained_on,
            "sample_count": self.sample_count,
            "rf_accuracy":  self.rf_acc,
            "gb_accuracy":  self.gb_acc,
            "lr_accuracy":  self.lr_acc,
        }


# Singleton
_predictor = None
def get_predictor() -> StudentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = StudentPredictor()
    return _predictor
