import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

INCOME_MAP = {"low": 0, "middle": 1, "high": 2}
PASS_MARK  = 40

# How much each past level contributes (always sums to 30%)
# Key = current level, Value = list of (level_name, weight) oldest→newest
LEVEL_WEIGHTS = {
    "100": [],                                                        # no history
    "200": [("100L", 0.30)],
    "300": [("100L", 0.15), ("200L", 0.15)],
    "400": [("100L", 0.10), ("200L", 0.10), ("300L", 0.10)],
    "500": [("100L", 0.075),("200L", 0.075),("300L", 0.075),("400L", 0.075)],
}
LEVEL_KEYS = {
    "100L": ("l100_physics","l100_prog","l100_stat","l100_cgpa"),
    "200L": ("l200_physics","l200_prog","l200_stat","l200_cgpa"),
    "300L": ("l300_physics","l300_prog","l300_stat","l300_cgpa"),
    "400L": ("l400_physics","l400_prog","l400_stat","l400_cgpa"),
}


def compute_risk(data):
    score   = 0
    hrs     = float(data.get("study_hours", 8))
    income  = data.get("family_income", "middle")
    partjob = int(data.get("has_part_time_job", 0))
    mental  = int(data.get("mental_health", 5))
    inet    = int(data.get("has_internet", 1))
    carry   = int(data.get("carryover_subjects", 0))

    if hrs < 3:    score += 20
    elif hrs < 6:  score += 12
    elif hrs < 10: score += 5

    if carry >= 4:   score += 20
    elif carry >= 2: score += 12
    elif carry >= 1: score += 6

    if mental <= 2:   score += 15
    elif mental <= 4: score += 9
    elif mental <= 6: score += 4

    if income == "low":      score += 8
    elif income == "middle": score += 2
    if partjob: score += 8
    if not inet: score += 4

    # Include all past level avgs in risk
    level = str(data.get("level", "100"))
    for lbl, _ in LEVEL_WEIGHTS.get(level, []):
        pk, prk, psk, pgk = LEVEL_KEYS[lbl]
        phy  = float(data.get(pk, 50))
        prog = float(data.get(prk, 50))
        stat = float(data.get(psk, 50))
        cgpa = float(data.get(pgk, 2.5))
        avg  = (phy + prog + stat) / 3
        if avg < 40:    score += 12
        elif avg < 50:  score += 7
        elif avg < 60:  score += 3
        if cgpa < 1.5:  score += 8
        elif cgpa < 2.5:score += 3

    score = min(score, 100)
    lvl = "HIGH" if score >= 55 else "MEDIUM" if score >= 25 else "LOW"
    return lvl, score


def avg_to_prob(avg, cgpa):
    if avg >= 70:   p = 95
    elif avg >= 60: p = 82
    elif avg >= 50: p = 68
    elif avg >= 45: p = 52
    elif avg >= 40: p = 38
    else:           p = 18
    if cgpa >= 4.0:   p = min(100, p + 8)
    elif cgpa >= 3.0: p = min(100, p + 4)
    elif cgpa < 1.5:  p = max(0,   p - 10)
    elif cgpa < 2.0:  p = max(0,   p - 5)
    return p


def cgpa_prediction(pass_prob, level_data, level):
    ml_cgpa = 1.0 + (pass_prob / 100) * 4.0
    weights = LEVEL_WEIGHTS.get(level, [])
    if not weights:
        return round(min(5.0, max(0.5, ml_cgpa)), 2)
    personal_total = 0.0
    for lbl, w in weights:
        pk, prk, psk, pgk = LEVEL_KEYS[lbl]
        phy  = float(level_data.get(pk, 0))
        prog = float(level_data.get(prk, 0))
        stat = float(level_data.get(psk, 0))
        cgpa = float(level_data.get(pgk, 0))
        if (phy + prog + stat) == 0:
            continue
        avg = (phy + prog + stat) / 3
        hist_cgpa = (4.5 if avg>=70 else 4.0 if avg>=60 else 3.0 if avg>=50
                     else 2.0 if avg>=45 else 1.5 if avg>=40 else 0.8)
        personal_cgpa = cgpa if cgpa > 0 else hist_cgpa
        personal_total += (w / 0.30) * personal_cgpa   # normalise weight to 1.0
    blended = (0.70 * ml_cgpa) + (0.30 * personal_total)
    return round(min(5.0, max(0.5, blended)), 2)


class StudentPredictor:
    def __init__(self):
        self.rf = self.gb = self.lr = self.scaler = None
        self.is_trained = False
        self.trained_on = "none"
        self.sample_count = self.rf_acc = self.gb_acc = self.lr_acc = 0
        self._train_synthetic()

    def _feats(self, r):
        return [
            float(r.get("calc_score", 0)), float(r.get("physics_score", 0)),
            float(r.get("chem_score", 0)), float(r.get("prog_score", 0)),
            float(r.get("stat_score", 0)), float(r.get("attendance", 75)),
            float(r.get("study_hours", 8)),
            float(INCOME_MAP.get(r.get("family_income","middle"), 1)),
            float(r.get("has_part_time_job", 0)), float(r.get("mental_health", 5)),
            float(r.get("has_internet", 1)), float(r.get("carryover_subjects", 0)),
        ]

    def _generate_synthetic(self, n=1500):
        np.random.seed(42)
        rows, labels = [], []
        for _ in range(n):
            hrs     = np.random.uniform(1, 20)
            income  = np.random.choice(["low","middle","high"], p=[0.3,0.5,0.2])
            mental  = int(np.clip(np.random.normal(6,2), 1, 10))
            partjob = int(np.random.random() < 0.3)
            inet    = int(np.random.random() < 0.75)
            carry   = int(np.random.choice([0,1,2,3,4], p=[0.5,0.25,0.14,0.07,0.04]))
            att     = float(np.clip(np.random.normal(70,15), 40, 100))
            bonus   = ((hrs-8)*1.5 + (mental-5)*1.2 + (INCOME_MAP[income]-1)*1.5
                       - partjob*3.0 + (inet-0.5) - carry*2.0)
            sc = {s: float(np.clip(np.random.normal(58+bonus,15),10,100))
                  for s in ["calc_score","physics_score","chem_score","prog_score","stat_score"]}
            passed = all(v >= PASS_MARK for v in sc.values())
            rows.append({**sc,"attendance":att,"study_hours":hrs,"family_income":income,
                         "has_part_time_job":partjob,"mental_health":mental,
                         "has_internet":inet,"carryover_subjects":carry})
            labels.append(1 if passed else 0)
        return rows, labels

    def _fit(self, rows, labels):
        X = np.array([self._feats(r) for r in rows])
        y = np.array(labels)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        self.scaler = StandardScaler()
        Xtr_sc = self.scaler.fit_transform(Xtr)
        Xte_sc = self.scaler.transform(Xte)
        self.rf = RandomForestClassifier(n_estimators=150,max_depth=8,random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100,max_depth=5,random_state=42)
        self.lr = LogisticRegression(max_iter=1000,random_state=42)
        self.rf.fit(Xtr,ytr); self.gb.fit(Xtr,ytr); self.lr.fit(Xtr_sc,ytr)
        self.rf_acc = round(accuracy_score(yte,self.rf.predict(Xte))*100,2)
        self.gb_acc = round(accuracy_score(yte,self.gb.predict(Xte))*100,2)
        self.lr_acc = round(accuracy_score(yte,self.lr.predict(Xte_sc))*100,2)
        self.sample_count = len(rows)
        self.is_trained = True

    def _train_synthetic(self):
        rows, labels = self._generate_synthetic()
        self._fit(rows, labels)
        self.trained_on = "synthetic"
        print(f"  AI trained (synthetic) RF:{self.rf_acc}% GB:{self.gb_acc}% LR:{self.lr_acc}%")

    def train_on_real_data(self, db_rows):
        if len(db_rows) < 20: return False
        labels = [1 if r.get("actual_result","FAIL")=="PASS" else 0 for r in db_rows]
        self._fit(db_rows, labels)
        self.trained_on = f"real ({len(db_rows)} students)"
        print(f"  Retrained real n={len(db_rows)} RF:{self.rf_acc}% GB:{self.gb_acc}%")
        return True

    def predict(self, data):
        level = str(data.get("level", "100"))
        weights = LEVEL_WEIGHTS.get(level, [])

        # ── Best proxy physics/prog/stat scores for ML features ──
        # Use most recent level's scores if available
        best_phy = best_prog = best_stat = 58.0  # median default
        for lbl, _ in reversed(weights):
            pk, prk, psk, _ = LEVEL_KEYS[lbl]
            phy = float(data.get(pk, 0))
            prog = float(data.get(prk, 0))
            stat = float(data.get(psk, 0))
            if (phy + prog + stat) > 0:
                best_phy, best_prog, best_stat = phy, prog, stat
                break

        feats = np.array([[
            58.0, best_phy, 58.0, best_prog, best_stat, 72.0,
            float(data.get("study_hours", 8)),
            float(INCOME_MAP.get(data.get("family_income","middle"), 1)),
            float(data.get("has_part_time_job", 0)),
            float(data.get("mental_health", 5)),
            float(data.get("has_internet", 1)),
            float(data.get("carryover_subjects", 0)),
        ]])
        feats_sc = self.scaler.transform(feats)
        rf_p = self.rf.predict_proba(feats)[0]
        gb_p = self.gb.predict_proba(feats)[0]
        lr_p = self.lr.predict_proba(feats_sc)[0]
        ml_prob = (0.4*rf_p[1] + 0.4*gb_p[1] + 0.2*lr_p[1]) * 100   # 70% component

        # ── Personal history: split 30% equally across past levels ──
        personal_total = 0.0
        levels_used = []
        for lbl, w in weights:
            pk, prk, psk, pgk = LEVEL_KEYS[lbl]
            phy  = float(data.get(pk, 0))
            prog = float(data.get(prk, 0))
            stat = float(data.get(psk, 0))
            cgpa = float(data.get(pgk, 0))
            if (phy + prog + stat) == 0:
                # Not filled in — redistribute to ML
                continue
            avg = (phy + prog + stat) / 3
            pp  = avg_to_prob(avg, cgpa)
            personal_total += w * pp
            levels_used.append({"level": lbl, "avg": round(avg,1),
                                 "cgpa": cgpa, "prob": pp, "weight": round(w*100)})

        # Work out actual split
        personal_weight = sum(e["weight"]/100 for e in levels_used) if levels_used else 0
        ml_weight = 1.0 - personal_weight   # at least 0.70, up to 1.0

        ens_pass = round((ml_weight * ml_prob) + personal_total, 1)

        # ── Hard override rules ───────────────────────────────────
        mental = int(data.get("mental_health", 5))
        carry  = int(data.get("carryover_subjects", 0))
        hrs    = float(data.get("study_hours", 8))
        inet   = int(data.get("has_internet", 1))
        income = data.get("family_income", "middle")
        risk_level, risk_score = compute_risk(data)

        if carry >= 5:                                 ens_pass = min(ens_pass, 20.0)
        if carry >= 4 and mental <= 3:                 ens_pass = min(ens_pass, 15.0)
        if hrs < 2 and carry >= 3:                     ens_pass = min(ens_pass, 18.0)
        if mental <= 1 and not inet and income=="low": ens_pass = min(ens_pass, 10.0)
        if risk_score >= 75:                           ens_pass = min(ens_pass, 40.0)
        if risk_score >= 90:                           ens_pass = min(ens_pass, 12.0)
        # Any past level avg below 40 is a serious penalty
        for e in levels_used:
            if e["avg"] < 40: ens_pass = min(ens_pass, 30.0)

        ens_pass = round(max(0, min(100, ens_pass)), 1)
        ens_fail = round(100 - ens_pass, 1)
        pred = "PASS" if ens_pass >= 50 else "FAIL"
        conf = "HIGH" if (ens_pass>=80 or ens_pass<=20) else "MEDIUM" if (ens_pass>=65 or ens_pass<=35) else "LOW"

        return {
            "predicted_result":   pred,
            "predicted_cgpa":     cgpa_prediction(ens_pass, data, level),
            "pass_probability":   ens_pass,
            "fail_probability":   ens_fail,
            "risk_level":         risk_level,
            "risk_score":         risk_score,
            "model_confidence":   conf,
            "ml_weight":          round(ml_weight * 100, 1),
            "personal_weight":    round(personal_weight * 100, 1),
            "ml_contribution":    round(ml_weight * ml_prob, 1),
            "personal_contribution": round(personal_total, 1),
            "levels_used":        levels_used,
            "has_prev_data":      len(levels_used) > 0,
            "rf_pass":            round(rf_p[1]*100, 1),
            "gb_pass":            round(gb_p[1]*100, 1),
            "lr_pass":            round(lr_p[1]*100, 1),
            "trained_on":         self.trained_on,
            "rf_acc":             self.rf_acc,
            "gb_acc":             self.gb_acc,
            "lr_acc":             self.lr_acc,
        }

    def status_info(self):
        return {"is_trained":self.is_trained,"trained_on":self.trained_on,
                "sample_count":self.sample_count,"rf_accuracy":self.rf_acc,
                "gb_accuracy":self.gb_acc,"lr_accuracy":self.lr_acc}

_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = StudentPredictor()
    return _predictor
