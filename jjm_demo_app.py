# jjm_demo_app.py
# Streamlit demo: photo upload -> OCR reading -> store in SQLite -> dashboard with roles

import streamlit as st
from PIL import Image
import io, re, os, datetime
import numpy as np
import pandas as pd
import pytesseract
import cv2
from sqlalchemy import create_engine, text

# --- Display Logo and Header ---
st.image("logo.jpg", width=180)
st.markdown("<h2 style='text-align:center; color:#0077b6;'>Jal Jeevan Mission – Assam Dashboard</h2>", unsafe_allow_html=True)
st.markdown("---")


# ---------- Simple DB (SQLite) ----------
DB_FILE = "jjm_demo.sqlite"
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
with engine.connect() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        division TEXT NOT NULL,
        meter_id TEXT,
        reading REAL,
        photo BLOB,
        uploaded_by TEXT,
        notes TEXT
    )"""))

# ---------- Helper: image preprocess + OCR ----------
def preprocess_and_ocr(pil_image):
    # Convert PIL -> OpenCV grayscale
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize small images to improve OCR
    h, w = gray.shape
    scale = max(1, 1024 / max(h, w))
    if scale > 1:
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    # Basic denoise and threshold
    gray = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: try adaptive threshold if OTSU fails
    # th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # Config: only digits and punctuation
    custom_oem_psm_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
    text = pytesseract.image_to_string(th, config=custom_oem_psm_config)
    return text, th

def extract_number(text):
    # Find the largest numeric token (handles decimals)
    candidates = re.findall(r'\d+(?:\.\d+)?', text)
    if not candidates:
        return None
    # Choose the longest (or highest) candidate as likely reading
    cand = max(candidates, key=lambda x: (len(x), float(x)))
    try:
        return float(cand)
    except:
        return None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="JJM Assam — Meter Photo Demo", layout="wide")
st.title("Jal Jeevan Mission Assam — Meter Photo → Dashboard (Demo)")

# Left column: upload form
left, right = st.columns([1,2])
with left:
    st.header("Upload meter photo")
    division = st.text_input("Division (e.g. Guwahati)", value="Guwahati")
    meter_id = st.text_input("Meter ID (optional)", value="MTR-001")
    uploader = st.text_input("Uploaded by (name/role)", value="Section Officer")
    notes = st.text_area("Notes (optional)", value="")
    uploaded_file = st.file_uploader("Choose a photo (jpg/png)", type=["jpg","jpeg","png"])
    if st.button("Process & Save"):
        if not uploaded_file:
            st.warning("Please upload a photo first.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded photo", use_column_width=True)
            ocr_text, preproc = preprocess_and_ocr(image)
            st.subheader("OCR raw text (for debugging)")
            st.code(ocr_text.strip() or "<no text recognized>")
            reading_val = extract_number(ocr_text)
            if reading_val is None:
                st.error("Could not detect a numeric reading. Try cropping the photo or improving lighting.")
            else:
                st.success(f"Detected reading: {reading_val}")
                # Save to DB (photo saved as bytes)
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                photo_bytes = buf.getvalue()
                ts = datetime.datetime.now().isoformat()
                with engine.begin() as conn:
                    conn.execute(text(
                        "INSERT INTO readings (timestamp, division, meter_id, reading, photo, uploaded_by, notes) VALUES (:timestamp,:division,:meter_id,:reading,:photo,:uploaded_by,:notes)"
                    ), {"timestamp": ts, "division": division, "meter_id": meter_id, "reading": reading_val, "photo": photo_bytes, "uploaded_by": uploader, "notes": notes})
                st.info("Saved to demo database.")

with right:
    st.header("Dashboard (Demo views)")
    # simulate role-based view
    role = st.selectbox("View as role", ["Section Officer","Assistant Executive Engineer","Executive Engineer","District Commissioner","Headquarters"])
    # date range
    today = datetime.date.today()
    default_from = (today - datetime.timedelta(days=6)).isoformat()
    date_from = st.date_input("From", value=datetime.date.fromisoformat(default_from))
    date_to = st.date_input("To", value=today)

    # Load data for range
    q = text("SELECT id,timestamp,division,meter_id,reading,uploaded_by,notes FROM readings WHERE date(timestamp) BETWEEN :dfrom AND :dto")
    with engine.connect() as conn:
        rows = conn.execute(q, {"dfrom": date_from.isoformat(), "dto": date_to.isoformat()}).fetchall()
    df = pd.DataFrame(rows, columns=["id","timestamp","division","meter_id","reading","uploaded_by","notes"])
    if df.empty:
        st.info("No readings found for selected range. Upload some photos to demonstrate.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        # Role-based aggregation logic (simplified)
        # Section Officer: see only their own uploads / division's raw readings
        # AEE/EE: see division-level aggregate
        # DC: see district-level (we'll treat division as district for demo)
        # HQ: see all divisions
        if role == "Section Officer":
            # show detail table
            st.subheader("Daily readings (detailed)")
            st.dataframe(df.sort_values("timestamp").reset_index(drop=True))
        elif role in ["Assistant Executive Engineer","Executive Engineer"]:
            st.subheader("Division summary (daily totals)")
            # group by date and division
            summary = df.groupby(["date","division"]).reading.agg(["count","sum","mean"]).reset_index()
            st.dataframe(summary)
            # small chart
            chart = summary.pivot(index="date", columns="division", values="sum").fillna(0)
            st.line_chart(chart)
        elif role == "District Commissioner":
            st.subheader("Weekly totals by division")
            weekly = df.groupby("division").reading.sum().reset_index().rename(columns={"reading":"weekly_total_litre"})
            st.table(weekly)
            st.bar_chart(weekly.set_index("division"))
        else: # HQ
            st.subheader("HQ Overview (absentees, totals)")
            # total per division
            totals = df.groupby("division").reading.sum().reset_index().rename(columns={"reading":"total_litre"})
            st.table(totals)
            # absentees: divisions with no reading today (for demo we define a small fixed list)
            all_divisions = ["Guwahati","Tezpur","Dibrugarh","Silchar"]
            present = set(df[df["date"]==today]["division"].unique())
            absentees = [d for d in all_divisions if d not in present]
            st.markdown(f"**Absentees (no reading today):** {', '.join(absentees) if absentees else 'None'}")

    st.markdown("---")
    st.write("Tip: use the upload panel to add sample readings. The dashboard updates immediately.")
