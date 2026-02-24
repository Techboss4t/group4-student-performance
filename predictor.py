
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

def compute_risk(data):
    score = 0
    hrs     = float(data.get("study_hours", 8))
    income  = data.get("family_income", "middle")
    partjob = int(data.get("has_part_time_job", 0))
    mental  = int(data.get("mental_health", 5))
    inet    = int(data.get("has_internet", 1))
    carry   = int(data.get("carryover_subjects", 0))
    prev_phy  = float(data.get("prev_physics_score", 50))
    prev_prog = float(data.get("prev_prog_score", 50))
    prev_stat = float(data.get("prev_stat_score", 50))
    prev_cgpa = float(data.get("prev_cgpa", 2.5))
    prev_avg  = (prev_phy + prev_prog + prev_stat) / 3
    if hrs < 3:    score += 20
    elif hrs < 6:  score += 12
    elif hrs < 10: score += 5
    if carry >= 4:   score += 20
    elif carry >= 2: score += 12
    elif carry >= 1: score += 6
    if mental <= 2:   score += 15
    elif mental <= 4: score += 9
    elif mental <= 6: score += 4
    if prev_avg < 40:    score += 25
    elif prev_avg < 50:  score += 15
    elif prev_avg < 60:  score += 8
    if prev_cgpa < 1.5:  score += 10
    elif prev_cgpa < 2.5:score += 5
    if income == "low":    score += 8
    elif income == "middle": score += 2
    if partjob: score += 8
    if not inet: score += 4
    score = min(score, 100)
    level = "HIGH" if score >= 55 else "MEDIUM" if score >= 25 else "LOW"
    return level, score

def cgpa_prediction(pass_prob, prev_cgpa, prev_avg):
    ml_cgpa = 1.0 + (pass_prob / 100) * 4.0
    if prev_avg >= 70:   hist = 4.5
    elif prev_avg >= 60: hist = 4.0
    elif prev_avg >= 50: hist = 3.0
    elif prev_avg >= 45: hist = 2.0
    elif prev_avg >= 40: hist = 1.5
    else:                hist = 0.8
    personal = prev_cgpa if prev_cgpa > 0 else hist
    return round(min(5.0, max(0.5, (0.70 * ml_cgpa) + (0.30 * personal))), 2)

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
            bonus   = (hrs-8)*1.5 + (mental-5)*1.2 + (INCOME_MAP[income]-1)*1.5 - partjob*3.0 + (inet-0.5) - carry*2.0
            sc = {s: float(np.clip(np.random.normal(58+bonus,15),10,100))
                  for s in ["calc_score","physics_score","chem_score","prog_score","stat_score"]}
            passed = all(v >= PASS_MARK for v in sc.values())
            rows.append({**sc,"attendance":att,"study_hours":hrs,"family_income":income,
                         "has_part_time_job":partjob,"mental_health":mental,"has_internet":inet,"carryover_subjects":carry})
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
        """
        Weighted prediction using ALL available level history:
          100L  → 100% AI
          200L  → 70% AI + 30% (100L)
          300L  → 70% AI + 20% (200L) + 10% (100L)
          400L  → 70% AI + 15% (300L) + 10% (200L) + 5% (100L)
          500L  → 70% AI + 12% (400L) + 10% (300L) + 5% (200L) + 3% (100L)
        """
        level = int(data.get("level", 100))

        # Helper: convert 3-subject avg + cgpa → pass probability
        def score_to_prob(phy, prog, stat, cgpa):
            if (phy + prog + stat) == 0:
                return None   # no data entered
            avg = (phy + prog + stat) / 3
            if avg >= 70:   pp = 95
            elif avg >= 60: pp = 82
            elif avg >= 50: pp = 68
            elif avg >= 45: pp = 52
            elif avg >= 40: pp = 38
            else:            pp = 18
            if cgpa >= 4.0:            pp = min(100, pp + 8)
            elif cgpa >= 3.0:          pp = min(100, pp + 4)
            elif 0 < cgpa < 1.5:       pp = max(0,   pp - 10)
            return pp, avg

        # ── Gather all previous level inputs ─────────────────────
        # prev  = most recent previous level  (e.g. 200L for 300L student)
        # prev2 = one before that             (e.g. 100L for 300L student)
        # prev3 = two before that             (e.g. 100L for 400L student)

        def get_lvl(prefix):
            phy  = float(data.get(f"{prefix}_physics_score", 0))
            prog = float(data.get(f"{prefix}_prog_score", 0))
            stat = float(data.get(f"{prefix}_stat_score", 0))
            cgpa = float(data.get(f"{prefix}_cgpa", 0))
            return phy, prog, stat, cgpa

        l1_phy, l1_prog, l1_stat, l1_cgpa = get_lvl("prev")    # most recent prev level
        l2_phy, l2_prog, l2_stat, l2_cgpa = get_lvl("prev2")   # 2 levels back
        l3_phy, l3_prog, l3_stat, l3_cgpa = get_lvl("prev3")   # 3 levels back

        # Compute personal probabilities per level
        r1 = score_to_prob(l1_phy, l1_prog, l1_stat, l1_cgpa)
        r2 = score_to_prob(l2_phy, l2_prog, l2_stat, l2_cgpa)
        r3 = score_to_prob(l3_phy, l3_prog, l3_stat, l3_cgpa)

        p1_prob, p1_avg = (r1[0], r1[1]) if r1 else (None, 0)
        p2_prob, p2_avg = (r2[0], r2[1]) if r2 else (None, 0)
        p3_prob, p3_avg = (r3[0], r3[1]) if r3 else (None, 0)

        # Best available avg for ML feature injection
        best_phy  = l1_phy  if l1_phy  > 0 else (l2_phy  if l2_phy  > 0 else 58.0)
        best_prog = l1_prog if l1_prog > 0 else (l2_prog if l2_prog > 0 else 58.0)
        best_stat = l1_stat if l1_stat > 0 else (l2_stat if l2_stat > 0 else 58.0)
        best_cgpa = l1_cgpa if l1_cgpa > 0 else (l2_cgpa if l2_cgpa > 0 else 0)

        # ── ML component (always 70%) ─────────────────────────────
        feats = np.array([[
            58.0, best_phy, 58.0, best_prog, best_stat,
            72.0,
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
        ml_prob = (0.4*rf_p[1] + 0.4*gb_p[1] + 0.2*lr_p[1]) * 100

        # ── Level-based weighting ─────────────────────────────────
        # Weights: (AI, prev1, prev2, prev3)
        WEIGHTS = {
            100: (1.00, 0,    0,    0   ),  # 100L: 100% AI
            200: (0.70, 0.30, 0,    0   ),  # 200L: 70% AI + 30% (100L)
            300: (0.70, 0.20, 0.10, 0   ),  # 300L: 70% AI + 20% (200L) + 10% (100L)
            400: (0.70, 0.15, 0.10, 0.05),  # 400L: 70% AI + 15% (300L) + 10% (200L) + 5% (100L)
            500: (0.70, 0.12, 0.10, 0.08),  # 500L: similar
        }
        w = WEIGHTS.get(level, WEIGHTS[200])
        ai_w, w1, w2, w3 = w

        # Build weighted blend — only use weights where data exists
        total_w  = ai_w
        ens_pass = ai_w * ml_prob
        personal_total = 0

        if p1_prob is not None and w1 > 0:
            ens_pass     += w1 * p1_prob
            total_w      += w1
            personal_total += w1 * p1_prob
        if p2_prob is not None and w2 > 0:
            ens_pass     += w2 * p2_prob
            total_w      += w2
            personal_total += w2 * p2_prob
        if p3_prob is not None and w3 > 0:
            ens_pass     += w3 * p3_prob
            total_w      += w3
            personal_total += w3 * p3_prob

        # Normalise if some levels were missing
        if total_w > 0:
            ens_pass = (ens_pass / total_w) * 100
        else:
            ens_pass = ml_prob

        ens_pass = round(ens_pass, 1)

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
        if p1_prob is not None and p1_avg < 40:        ens_pass = min(ens_pass, 35.0)

        ens_pass = round(max(0, min(100, ens_pass)), 1)
        ens_fail = round(100 - ens_pass, 1)
        pred = "PASS" if ens_pass >= 50 else "FAIL"
        conf = "HIGH" if (ens_pass>=80 or ens_pass<=20) else "MEDIUM" if (ens_pass>=65 or ens_pass<=35) else "LOW"

        # Cumulative CGPA from all levels
        all_cgpas = [c for c in [l1_cgpa, l2_cgpa, l3_cgpa] if c > 0]
        avg_hist_cgpa = sum(all_cgpas)/len(all_cgpas) if all_cgpas else 0
        best_avg = p1_avg if p1_avg > 0 else (p2_avg if p2_avg > 0 else 0)
        pred_cgpa = cgpa_prediction(ens_pass, avg_hist_cgpa, best_avg)

        has_any_prev = any(x is not None for x in [p1_prob, p2_prob, p3_prob])

        return {
            "predicted_result":      pred,
            "predicted_cgpa":        pred_cgpa,
            "pass_probability":      ens_pass,
            "fail_probability":      ens_fail,
            "risk_level":            risk_level,
            "risk_score":            risk_score,
            "model_confidence":      conf,
            "ml_contribution":       round(ai_w * ml_prob, 1),
            "personal_contribution": round(personal_total, 1),
            "has_prev_data":         has_any_prev,
            "prev_avg":              round(best_avg, 1),
            "rf_pass":               round(rf_p[1]*100, 1),
            "gb_pass":               round(gb_p[1]*100, 1),
            "lr_pass":               round(lr_p[1]*100, 1),
            "trained_on":            self.trained_on,
            "rf_acc":                self.rf_acc,
            "gb_acc":                self.gb_acc,
            "lr_acc":                self.lr_acc,
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
