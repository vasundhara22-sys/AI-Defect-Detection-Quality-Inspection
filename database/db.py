import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
import os

DB_PATH = Path(os.environ.get("DEFECT_DB", "database/inspections.db"))

def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(_conn()) as c, c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                product_id TEXT,
                image_file TEXT NOT NULL,
                system_status TEXT NOT NULL,
                top_confidence REAL,
                detections TEXT,
                review_decision TEXT,
                reviewer TEXT,
                review_note TEXT,
                reviewed_at TEXT,
                final_status TEXT NOT NULL
            )""")


def add_inspection(product_id, image_file, status, detections):
    top = max((d["confidence"] for d in detections), default=0.0)
    with closing(_conn()) as c, c:
        cur = c.execute(
            """INSERT INTO inspections
               (timestamp, product_id, image_file, system_status, top_confidence,
                detections, final_status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(timespec="seconds"), product_id, image_file,
             status, top, json.dumps(detections), status),
        )
        return cur.lastrowid


def _row(r):
    d = dict(r)
    d["detections"] = json.loads(d["detections"] or "[]")
    return d


def get_inspection(inspection_id):
    with closing(_conn()) as c:
        r = c.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
    return _row(r) if r else None


def list_inspections(limit=50, status=None):
    sql, args = "SELECT * FROM inspections", []
    if status:
        sql += " WHERE final_status = ?"
        args.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with closing(_conn()) as c:
        return [_row(r) for r in c.execute(sql, args).fetchall()]


def add_review(inspection_id, decision, reviewer, note):
    with closing(_conn()) as c, c:
        cur = c.execute(
            """UPDATE inspections
               SET review_decision = ?, reviewer = ?, review_note = ?,
                   reviewed_at = ?, final_status = ?
               WHERE id = ?""",
            (decision, reviewer, note, datetime.now().isoformat(timespec="seconds"),
             decision, inspection_id),
        )
        return cur.rowcount > 0


def stats():
    with closing(_conn()) as c:
        rows = [_row(r) for r in c.execute("SELECT * FROM inspections").fetchall()]
    total = len(rows)
    by_status, by_defect, by_day = {}, {}, {}
    for r in rows:
        by_status[r["final_status"]] = by_status.get(r["final_status"], 0) + 1
        for cls in {d["class"] for d in r["detections"]}:
            by_defect[cls] = by_defect.get(cls, 0) + 1
        day = r["timestamp"][:10]
        by_day.setdefault(day, {"total": 0, "fail": 0})
        by_day[day]["total"] += 1
        by_day[day]["fail"] += r["final_status"] == "Fail"
    return {
        "total": total,
        "by_status": by_status,
        "failure_rate": round(by_status.get("Fail", 0) / total, 3) if total else 0,
        "pending_review": sum(1 for r in rows
                              if r["system_status"] == "Review Required" and not r["review_decision"]),
        "by_defect": by_defect,
        "by_day": by_day,
    }