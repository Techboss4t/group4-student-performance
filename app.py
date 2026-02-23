"""
app.py v5.0 — Flask Backend
Group 4 Student Performance Prediction System

Two clear phases:
  /api/train   — upload CSV of past students → trains AI
  /api/predict — enter lifestyle data → AI predicts future
"""

import os, io, csv, json
from flask import Flask, request, jsonify, send_from_directory

from database import (init_db, insert_training_batch, get_all_training_data,
                      get_training_count, save_prediction, get_predictions,
                      delete_prediction, get_dashboard_stats, log_training)
from predictor import get_predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

print("\n" + "="*60)
print("  GROUP 4 — Student Performance Prediction System v5.0")
print("  Phase 1: Train AI  |  Phase 2: Predict Future")
print("="*60)
init_db()
predictor = get_predictor()

# Retrain on real data at startup if enough exists
real = get_all_training_data()
if len(real) >= 20:
    predictor.train_on_real_data(real)
    print(f"  ✔  Retrained on {len(real)} real students at startup")


# ── Serve Frontend ────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# ── Dashboard Stats ───────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    stats = get_dashboard_stats()
    stats.update(predictor.status_info())
    return jsonify(stats)


# ── PHASE 1: Upload Training CSV ──────────────────────────────
@app.route("/api/train", methods=["POST"])
def api_train():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    text   = request.files["file"].read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    batch, errors, skipped = [], [], 0

    for i, row in enumerate(reader, 1):
        row = {k.strip().lower().replace(" ","_"): v.strip() for k,v in row.items()}
        try:
            # Determine actual result
            result = row.get("result", row.get("actual_result", row.get("status", ""))).upper()
            if result not in ("PASS", "FAIL"):
                # Try to infer from CGPA if result column missing
                cgpa = float(row.get("cgpa", row.get("actual_cgpa", 0)) or 0)
                result = "PASS" if cgpa >= 1.5 else "FAIL"

            batch.append({
                "name":             row.get("name", f"Student {i}"),
                "matric_no":        row.get("matric_no", row.get("matric", f"UNK/{i:04d}")),
                "calc_score":       float(row.get("calc_score", row.get("calculus_score", 0)) or 0),
                "physics_score":    float(row.get("physics_score", 0) or 0),
                "chem_score":       float(row.get("chem_score", row.get("chemistry_score", 0)) or 0),
                "prog_score":       float(row.get("prog_score", row.get("programming_score", 0)) or 0),
                "stat_score":       float(row.get("stat_score", row.get("statistics_score", 0)) or 0),
                "attendance":       float(row.get("attendance", 75) or 75),
                "study_hours":      float(row.get("study_hours", 8) or 8),
                "family_income":    row.get("family_income", "middle") or "middle",
                "has_part_time_job":int(float(row.get("has_part_time_job", 0) or 0)),
                "mental_health":    int(float(row.get("mental_health", 5) or 5)),
                "has_internet":     int(float(row.get("has_internet", 1) or 1)),
                "carryover_subjects":int(float(row.get("carryover_subjects", 0) or 0)),
                "actual_result":    result,
                "actual_cgpa":      float(row.get("cgpa", row.get("actual_cgpa", 0)) or 0),
            })
        except Exception as e:
            errors.append({"row": i, "error": str(e)}); skipped += 1

    inserted = insert_training_batch(batch) if batch else 0

    # Retrain AI on ALL real data now
    all_real = get_all_training_data()
    retrained = False
    if len(all_real) >= 20:
        retrained = predictor.train_on_real_data(all_real)
        if retrained:
            log_training(predictor.sample_count, predictor.rf_acc,
                         predictor.gb_acc, predictor.lr_acc)

    return jsonify({
        "imported":    inserted,
        "skipped":     skipped,
        "errors":      errors[:10],
        "total_training": len(all_real),
        "retrained":   retrained,
        "rf_accuracy": predictor.rf_acc,
        "gb_accuracy": predictor.gb_acc,
        "lr_accuracy": predictor.lr_acc,
    })


# ── PHASE 2: Predict One Student ──────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No data sent"}), 400

    # Run prediction
    result = predictor.predict(body)

    # Save to DB
    row_id = save_prediction({
        "name":              body.get("name", "Unknown"),
        "matric_no":         body.get("matric_no", "N/A"),
        "level":             body.get("level", "100"),
        "study_hours":       float(body.get("study_hours", 8)),
        "family_income":     body.get("family_income", "middle"),
        "has_part_time_job": int(body.get("has_part_time_job", 0)),
        "mental_health":     int(body.get("mental_health", 5)),
        "has_internet":      int(body.get("has_internet", 1)),
        "carryover_subjects":int(body.get("carryover_subjects", 0)),
        **result,
    })

    return jsonify({"id": row_id, **result})


# ── Get All Predictions ───────────────────────────────────────
@app.route("/api/predictions")
def api_get_predictions():
    rows, total = get_predictions(
        search   = request.args.get("search",""),
        result   = request.args.get("result","all"),
        risk     = request.args.get("risk","all"),
        sort     = request.args.get("sort","id"),
        order    = request.args.get("order","DESC"),
        page     = int(request.args.get("page",1)),
        per_page = int(request.args.get("per_page",20)),
    )
    per_page = int(request.args.get("per_page",20))
    return jsonify({
        "predictions": rows, "total": total,
        "page": int(request.args.get("page",1)),
        "pages": (total+per_page-1)//per_page,
    })


# ── Delete Prediction ─────────────────────────────────────────
@app.route("/api/predictions/<int:pid>", methods=["DELETE"])
def api_delete_pred(pid):
    delete_prediction(pid)
    return jsonify({"deleted": pid})


# ── Model Status ──────────────────────────────────────────────
@app.route("/api/model")
def api_model():
    return jsonify(predictor.status_info())


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("RENDER") is None
    print(f"\n  Open browser: http://127.0.0.1:{port}\n")
    app.run(debug=debug, host="0.0.0.0", port=port)
