"""
database.py v5.0
Two separate tables:
  training_data  — past students with known results (used to train AI)
  predictions    — new students predicted by AI (no scores, lifestyle only)
"""
import sqlite3, os, json

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "group4.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

SCHEMA = """
-- ── TABLE 1: Training Data (past students with known results) ──
CREATE TABLE IF NOT EXISTS training_data (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT,
    matric_no        TEXT,

    -- Subject scores (past known data)
    calc_score       REAL DEFAULT 0,
    physics_score    REAL DEFAULT 0,
    chem_score       REAL DEFAULT 0,
    prog_score       REAL DEFAULT 0,
    stat_score       REAL DEFAULT 0,

    -- Lifestyle factors
    attendance       REAL DEFAULT 0,
    study_hours      REAL DEFAULT 0,
    family_income    TEXT DEFAULT 'middle',
    has_part_time_job INTEGER DEFAULT 0,
    mental_health    INTEGER DEFAULT 5,
    has_internet     INTEGER DEFAULT 1,
    carryover_subjects INTEGER DEFAULT 0,

    -- Actual known result (this is what AI learns from)
    actual_result    TEXT NOT NULL,   -- PASS or FAIL
    actual_cgpa      REAL DEFAULT 0,

    imported_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── TABLE 2: Predictions (new students — no scores yet) ──────
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    matric_no        TEXT NOT NULL UNIQUE,
    level            TEXT DEFAULT '100',

    -- Lifestyle inputs ONLY (no scores)
    study_hours      REAL DEFAULT 0,
    family_income    TEXT DEFAULT 'middle',
    has_part_time_job INTEGER DEFAULT 0,
    mental_health    INTEGER DEFAULT 5,
    has_internet     INTEGER DEFAULT 1,
    carryover_subjects INTEGER DEFAULT 0,

    -- AI prediction outputs
    predicted_result TEXT,     -- PASS or FAIL
    predicted_cgpa   REAL,
    pass_probability REAL,
    risk_level       TEXT,     -- LOW / MEDIUM / HIGH
    risk_score       INTEGER,
    model_confidence TEXT,     -- HIGH / MEDIUM / LOW

    predicted_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── TABLE 3: Model training log ──────────────────────────────
CREATE TABLE IF NOT EXISTS model_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_count INTEGER,
    rf_accuracy  REAL,
    gb_accuracy  REAL,
    lr_accuracy  REAL,
    trained_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_td_result ON training_data(actual_result);
CREATE INDEX IF NOT EXISTS idx_pred_risk  ON predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_pred_matric ON predictions(matric_no);
"""

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    print(f"  ✔  Database ready: {DB_PATH}")

# ── Training Data ─────────────────────────────────────────────
def insert_training_batch(rows: list) -> int:
    sql = """
        INSERT OR IGNORE INTO training_data
            (name, matric_no, calc_score, physics_score, chem_score, prog_score, stat_score,
             attendance, study_hours, family_income, has_part_time_job, mental_health,
             has_internet, carryover_subjects, actual_result, actual_cgpa)
        VALUES
            (:name,:matric_no,:calc_score,:physics_score,:chem_score,:prog_score,:stat_score,
             :attendance,:study_hours,:family_income,:has_part_time_job,:mental_health,
             :has_internet,:carryover_subjects,:actual_result,:actual_cgpa)
    """
    with get_conn() as conn:
        conn.executemany(sql, rows)
    return len(rows)

def get_training_count():
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]

def get_all_training_data():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM training_data").fetchall()
    return [dict(r) for r in rows]

# ── Predictions ───────────────────────────────────────────────
def save_prediction(data: dict) -> int:
    sql = """
        INSERT INTO predictions
            (name, matric_no, level, study_hours, family_income, has_part_time_job,
             mental_health, has_internet, carryover_subjects,
             predicted_result, predicted_cgpa, pass_probability,
             risk_level, risk_score, model_confidence)
        VALUES
            (:name,:matric_no,:level,:study_hours,:family_income,:has_part_time_job,
             :mental_health,:has_internet,:carryover_subjects,
             :predicted_result,:predicted_cgpa,:pass_probability,
             :risk_level,:risk_score,:model_confidence)
        ON CONFLICT(matric_no) DO UPDATE SET
            level=excluded.level, study_hours=excluded.study_hours,
            family_income=excluded.family_income, has_part_time_job=excluded.has_part_time_job,
            mental_health=excluded.mental_health, has_internet=excluded.has_internet,
            carryover_subjects=excluded.carryover_subjects,
            predicted_result=excluded.predicted_result, predicted_cgpa=excluded.predicted_cgpa,
            pass_probability=excluded.pass_probability, risk_level=excluded.risk_level,
            risk_score=excluded.risk_score, model_confidence=excluded.model_confidence,
            predicted_at=CURRENT_TIMESTAMP
    """
    with get_conn() as conn:
        cur = conn.execute(sql, data)
        return cur.lastrowid

def get_predictions(search='', result='all', risk='all',
                    sort='id', order='DESC', page=1, per_page=20):
    where, params = [], []
    if search:
        where.append("(name LIKE ? OR matric_no LIKE ?)")
        params += [f'%{search}%', f'%{search}%']
    if result in ('PASS','FAIL'):
        where.append("predicted_result=?"); params.append(result)
    if risk in ('LOW','MEDIUM','HIGH'):
        where.append("risk_level=?"); params.append(risk)
    wh = 'WHERE ' + ' AND '.join(where) if where else ''
    sort  = sort  if sort  in ('id','name','predicted_cgpa','risk_score','pass_probability','predicted_at') else 'id'
    order = order if order in ('ASC','DESC') else 'DESC'
    offset = (page-1)*per_page
    with get_conn() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM predictions {wh}", params).fetchone()[0]
        rows  = conn.execute(
            f"SELECT * FROM predictions {wh} ORDER BY {sort} {order} LIMIT ? OFFSET ?",
            params+[per_page, offset]
        ).fetchall()
    return [dict(r) for r in rows], total

def delete_prediction(pid: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM predictions WHERE id=?", (pid,))

def get_dashboard_stats():
    with get_conn() as conn:
        train_count = conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
        train_pass  = conn.execute("SELECT COUNT(*) FROM training_data WHERE actual_result='PASS'").fetchone()[0]
        pred_count  = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        pred_pass   = conn.execute("SELECT COUNT(*) FROM predictions WHERE predicted_result='PASS'").fetchone()[0]
        high_risk   = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='HIGH'").fetchone()[0]
        med_risk    = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='MEDIUM'").fetchone()[0]
        low_risk    = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='LOW'").fetchone()[0]
        avg_pred_cgpa = conn.execute("SELECT AVG(predicted_cgpa) FROM predictions").fetchone()[0] or 0
    return {
        "training_count": train_count,
        "training_pass":  train_pass,
        "training_fail":  train_count - train_pass,
        "pred_count":     pred_count,
        "pred_pass":      pred_pass,
        "pred_fail":      pred_count - pred_pass,
        "high_risk":      high_risk,
        "medium_risk":    med_risk,
        "low_risk":       low_risk,
        "avg_predicted_cgpa": round(avg_pred_cgpa, 2),
    }

def log_training(count, rf, gb, lr):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO model_log (sample_count,rf_accuracy,gb_accuracy,lr_accuracy) VALUES (?,?,?,?)",
            (count, rf, gb, lr)
        )

if __name__ == "__main__":
    init_db()
    print(get_dashboard_stats())
