import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("DEFECT_API", "http://127.0.0.1:8000")
ICON = {"Pass": "🟢", "Fail": "🔴", "Review Required": "🟠"}

st.set_page_config(page_title="Defect Inspection", layout="wide")

# ---------- helpers ----------
try:
    requests.get(f"{API}/health", timeout=3).raise_for_status()
except Exception:
    st.error("Backend is not running. Start it in another terminal:\n\n"
             "uvicorn backend.main:app --reload")
    st.stop()


def api_get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def annotated_bytes(image_file):
    r = requests.get(f"{API}/files/{image_file}", timeout=30)
    return r.content if r.ok else None


def det_table(detections):
    if not detections:
        return pd.DataFrame(columns=["class", "confidence"])
    return pd.DataFrame([{"class": d["class"], "confidence": d["confidence"]} for d in detections])


st.sidebar.title("Defect Inspection")
st.sidebar.error("PROTOTYPE. Not validated for production. All results need human review.")
page = st.sidebar.radio("Go to", ["Inspect", "Video", "Review", "Dashboard", "History"])
st.sidebar.caption("Human review stays available for uncertain results.")

# ---------- Inspect ----------
if page == "Inspect":
    st.header("Inspect a product")
    products = requests.get(f"{API}/products", timeout=10).json()
    product_id = st.selectbox("Product", list(products), format_func=lambda k: products[k]["name"])
    st.caption(products[product_id]["status"])
    source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    if source == "Upload":
        file = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "bmp"])
    else:
        file = st.camera_input("Take a photo of the product")

    if file and st.button("Run inspection", type="primary"):
        with st.spinner("Analysing image..."):
            r = requests.post(
                f"{API}/inspect",
                files={"file": (getattr(file, "name", "capture.jpg"), file.getvalue(), "image/jpeg")},
                data={"product_id": product_id},
                timeout=120,
            )
        if not r.ok:
            st.error(f"Inspection failed: {r.text}")
        else:
            res = r.json()
            st.subheader(f"{ICON[res['status']]} {res['status']}  (inspection #{res['id']})")
            c1, c2 = st.columns(2)
            c1.image(file.getvalue(), caption="Original", use_container_width=True)
            c2.image(annotated_bytes(res["annotated_image"].split("/")[-1]),
                     caption="Detected defects", use_container_width=True)
            if res["detections"]:
                st.dataframe(det_table(res["detections"]), use_container_width=True)
            else:
                st.info("No defects detected.")
            if res["status"] == "Review Required":
                st.warning("Confidence is low. Please check it on the Review page.")

# ---------- Video ----------
elif page == "Video":
    st.header("Inspect a video")
    st.caption("Upload a conveyor video. Each product that stops in view is inspected like an uploaded image.")
    vfile = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    prefix = st.text_input("Product ID prefix (optional)", value="VID")
    mode = st.radio("How to pick frames", ["Detect stopped products", "Every N seconds"], horizontal=True)
    if mode == "Every N seconds":
        every_s = st.slider("Seconds between inspections", 0.5, 5.0, 1.0, 0.5)
    else:
        stable_thr = st.slider("Stillness needed (lower = must be stiller)", 0.2, 5.0, 0.5, 0.1)
    max_items = st.number_input("Maximum products to inspect", 1, 100, 20)
    use_roi = st.checkbox("Crop to the inspection area before analysis", value=True)
    if use_roi:
        r1, r2, r3, r4 = st.columns(4)
        rx = r1.number_input("Left %", 0, 90, 33)
        ry = r2.number_input("Top %", 0, 90, 21)
        rw = r3.number_input("Width %", 5, 100, 34)
        rh = r4.number_input("Height %", 5, 100, 58)

    def inspect_frame(frame, pid):
        ok, buf = cv2.imencode(".jpg", frame)
        r = requests.post(f"{API}/inspect",
                          files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                          data={"product_id": pid}, timeout=120)
        if not r.ok:
            st.error(f"Inspection failed: {r.text}")
            return None
        return r.json()

    if vfile and st.button("Run video inspection", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(vfile.getvalue())
            path = tmp.name

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, int(fps * (every_s if mode == "Every N seconds" else 0.2)))

        results, prev, idx = [], None, 0
        still_run, armed = 0, True
        bar = st.progress(0.0, text="Reading video...")
        while len(results) < max_items and cap.grab():
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    pick = False
                    if mode == "Every N seconds":
                        pick = True
                    else:
                        g = cv2.GaussianBlur(
                            cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY),
                            (5, 5), 0).astype(float)
                        d = None if prev is None else np.abs(g - prev).mean()
                        prev = g
                        if d is not None and d < stable_thr:
                            still_run += 1
                            if still_run >= 2 and armed:
                                pick = True
                                armed = False
                        else:
                            still_run = 0
                            armed = True
                    if pick:
                        n = len(results) + 1
                        frame_in = frame
                        if use_roi:
                            fh, fw = frame.shape[:2]
                            x0, y0 = int(fw * rx / 100), int(fh * ry / 100)
                            x1 = min(fw, x0 + int(fw * rw / 100))
                            y1 = min(fh, y0 + int(fh * rh / 100))
                            frame_in = frame[y0:y1, x0:x1]
                        res = inspect_frame(frame_in, "steel_surface")
                        if res is None:
                            break
                        results.append(res)
                        bar.progress(min(idx / total, 1.0), text=f"Inspected {n} product(s)...")
            idx += 1
        cap.release()
        os.remove(path)
        bar.progress(1.0, text="Done")

        if not results:
            st.warning("No products were found. Try 'Every N seconds', or raise the stillness slider.")
        else:
            counts = pd.Series([r["status"] for r in results]).value_counts()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Products inspected", len(results))
            k2.metric("Pass", int(counts.get("Pass", 0)))
            k3.metric("Fail", int(counts.get("Fail", 0)))
            k4.metric("Review Required", int(counts.get("Review Required", 0)))
            st.info("All results are saved. Check the Review, Dashboard and History pages.")

            cols = st.columns(3)
            for i, res in enumerate(results):
                with cols[i % 3]:
                    img = annotated_bytes(res["annotated_image"].split("/")[-1])
                    if img:
                        st.image(img, use_container_width=True,
                                 caption=f"{ICON[res['status']]} {res['status']} (inspection #{res['id']})")

# ---------- Review ----------
elif page == "Review":
    st.header("Review uncertain results")
    pending = api_get("/inspections", limit=200, status="Review Required")
    if not pending:
        st.success("Nothing waiting for review.")
    else:
        st.write(f"{len(pending)} waiting for review")
        chosen = st.selectbox(
            "Choose an inspection", pending,
            format_func=lambda r: f"#{r['id']}  |  {r['product_id'] or 'no ID'}  |  {r['timestamp']}",
        )
        c1, c2 = st.columns([2, 1])
        img = annotated_bytes(chosen["image_file"])
        if img:
            c1.image(img, caption="Detected defects", use_container_width=True)
        c2.write(f"**System result:** {ICON[chosen['system_status']]} {chosen['system_status']}")
        c2.write(f"**Top confidence:** {chosen['top_confidence']}")
        c2.dataframe(det_table(chosen["detections"]), use_container_width=True)

        with st.form("review_form"):
            decision = st.radio("Your decision", ["Pass", "Fail"], horizontal=True)
            reviewer = st.text_input("Your name")
            note = st.text_area("Note (optional)")
            submitted = st.form_submit_button("Submit review", type="primary")
        if submitted:
            if not reviewer.strip():
                st.warning("Please enter your name.")
            else:
                r = requests.post(
                    f"{API}/inspections/{chosen['id']}/review",
                    json={"decision": decision, "reviewer": reviewer.strip(), "note": note},
                    timeout=30,
                )
                if r.ok:
                    st.success("Review saved.")
                    st.rerun()
                else:
                    st.error(r.text)

# ---------- Dashboard ----------
elif page == "Dashboard":
    st.header("Quality dashboard")
    s = api_get("/stats")
    if s["total"] == 0:
        st.info("No inspections yet.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total inspected", s["total"])
        k2.metric("Failure rate", f"{s['failure_rate'] * 100:.1f}%")
        k3.metric("Passed", s["by_status"].get("Pass", 0))
        k4.metric("Pending review", s["pending_review"])

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Results by status")
            st.bar_chart(pd.Series(s["by_status"], name="count"))
        with c2:
            st.subheader("Defect categories")
            st.bar_chart(pd.Series(s["by_defect"], name="count"))

        st.subheader("Trend over time")
        days = pd.DataFrame(s["by_day"]).T
        days["failure_rate_%"] = (days["fail"] / days["total"] * 100).round(1)
        st.line_chart(days[["total", "fail"]])
        st.dataframe(days, use_container_width=True)

# ---------- History ----------
elif page == "History":
    st.header("Inspection history")
    flt = st.selectbox("Filter by final status", ["All", "Pass", "Fail", "Review Required"])
    rows = api_get("/inspections", limit=500, **({} if flt == "All" else {"status": flt}))
    if not rows:
        st.info("No records.")
    else:
        df = pd.DataFrame(rows)
        df["defects"] = df["detections"].apply(lambda d: ", ".join(sorted({x["class"] for x in d})))
        cols = ["id", "timestamp", "product_id", "system_status", "top_confidence",
                "defects", "review_decision", "reviewer", "review_note", "final_status"]
        st.dataframe(df[cols], use_container_width=True)
        st.download_button("Download report (CSV)", df[cols].to_csv(index=False),
                           file_name="inspection_report.csv", mime="text/csv")
