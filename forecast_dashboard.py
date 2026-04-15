
import os, io, json, time, base64, struct, joblib, requests, smtplib, ssl, math
import numpy as np, pandas as pd, geopandas as gpd, folium
from folium import Popup
from dotenv import load_dotenv
from plyer import notification
import streamlit as st
from email.mime.text import MIMEText

load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
OWM_API_KEY = os.getenv("OWM_API_KEY")
GEOJSON_PATH = "kelowna_regions2/kelowna_36_regions.geojson"
DATASET_PATH = "dataset.csv"
MODELS_DIR = "models2"
ARTIFACT_PATH = os.path.join(MODELS_DIR, "ensemble_artifact.json")
DEFAULT_CENTER = [49.88, -119.49]
DEFAULT_ZOOM = 6
HIGH_THRESH = 0.70
REFRESH_INTERVAL = 300

st.set_page_config(layout="wide", page_title="Kelowna Wildfire Risk Portal")


st.markdown("""
<style>

.block-container {
    max-width: 80%;
    margin: auto;
    text-align: center;
    padding-top: 1rem;
}


ul, ol, li, p {
    text-align: left;
    margin-left: auto;
    margin-right: auto;
    width: 80%;
    line-height: 1.6;
}

h1, h2, h3, h4, h5 {
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
}
.section-title {
    color: #b33a3a;
    font-size: 26px;
    margin-top: 30px;
    margin-bottom: 10px;
}
.divider {
    border-top: 1px solid #ddd;
    margin: 25px auto;
    width: 80%;
}

.legend-box {
    background: #f0f2f6;
    padding: 12px;
    border-radius: 6px;
    border-left: 4px solid #ff6f00;
    color: #000;
    font-size: 15px;
    line-height: 1.6;
    text-align: left;
    width: 70%;
    margin: 20px auto;
}
.contact-card {
    background: #b33a3a;
    color: white;
    border-radius: 10px;
    padding: 15px;
    margin: 20px auto;
    width: 60%;
    text-align: center;
}
.stButton > button {
    white-space: nowrap !important;      
    display: inline-flex !important;     
    align-items: center !important;      
    justify-content: center !important;
    gap: 6px !important;                 
    padding: 6px 16px !important;
    font-size: 14px !important;
}

iframe, .stButton, .stCheckbox {
    margin-left: auto !important;
    margin-right: auto !important;
    display: flex;
    justify-content: center;
}

[data-testid="column"] {
    justify-content: center;
    align-items: start;
}
</style>
""", unsafe_allow_html=True)


ss = st.session_state
for k, v in {
    "last_refresh": 0.0, "weather": {}, "preds": {},
    "last_time": "N/A", "auto_refresh": True, "force_refresh": False, "forecast_hour": 0, "forecast_mode": False
}.items():
    if k not in ss: ss[k] = v

forecast_hour = int(ss.get("forecast_hour", 0))
forecast_mode = bool(ss.get("forecast_mode", False)) or (forecast_hour > 0)

from streamlit_autorefresh import st_autorefresh

if ss.auto_refresh:
    st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh_timer")


artifact = json.load(open(ARTIFACT_PATH))
model_names = artifact["model_names"]
FEATURES = artifact["features"]
artifact_type = artifact.get("type", "weighted")

weights, meta_model, meta_scaler = None, None, None
if artifact_type == "weighted":
    weights = np.array(artifact["weights"], dtype=float)
elif artifact_type == "stacking":
    meta_model_path = artifact.get("meta_model_path")
    meta_scaler_path = artifact.get("meta_scaler_path")
    if meta_model_path and os.path.exists(meta_model_path):
        meta_model = joblib.load(meta_model_path)
    if meta_scaler_path and os.path.exists(meta_scaler_path):
        meta_scaler = joblib.load(meta_scaler_path)

def safe_load(path):
    try: return joblib.load(path)
    except: return None

base_models = {}
for n in model_names:
    p1, p2 = os.path.join(MODELS_DIR, f"{n}_calibrated.pkl"), os.path.join(MODELS_DIR, f"{n}.pkl")
    m = safe_load(p1 if os.path.exists(p1) else p2)
    if m is not None: base_models[n] = m

def predict_proba_safe(m, X):
    try: return m.predict_proba(X.values)[:, 1]
    except: return m.predict(X.values).astype(float)

def color_for_prob(p):
    p = max(0.0, min(1.0, float(p)))
    if p < 0.33: r, g = int(255*p/0.33), 255
    elif p < 0.66: r, g = 255, int(255-85*(p-0.33)/0.33)
    else: r, g = 255, int(170-170*(p-0.66)/0.34)
    return f"#{r:02x}{g:02x}00"

def play_alarm():
    mp3_path = "siren.mp3"
    with open(mp3_path, "rb") as f:
        mp3_bytes = f.read()
    b64 = base64.b64encode(mp3_bytes).decode()
    st.markdown(f"""<audio autoplay style="display:none"><source src="data:audio/mp3;base64,{b64}"type="audio/mp3"></audio>""", unsafe_allow_html=True)

def send_email_alert(region_id, prob):
    if not EMAIL_USER or not EMAIL_PASS: return
    msg = MIMEText(f"High wildfire risk detected!\n\nRegion: {region_id}\nRisk Probability: {prob:.2f}\nStay alert!")
    msg["Subject"] = f"🔥 Wildfire Risk Alert — Region {region_id}"
    msg["From"], msg["To"] = EMAIL_USER, EMAIL_USER
    try:
        s = smtplib.SMTP("smtp.gmail.com", 587); s.starttls()
        s.login(EMAIL_USER, EMAIL_PASS)
        s.sendmail(EMAIL_USER, EMAIL_USER, msg.as_string()); s.quit()
    except: pass

def send_windows_alert(region_id, prob):
    try:
        notification.notify(
            title=f"🔥 Fire Risk Alert — Region {region_id}",
            message=f"Predicted probability: {prob:.2f}",
            timeout=6
        )
    except: pass

gdf = gpd.read_file(GEOJSON_PATH)
df = pd.read_csv(DATASET_PATH).sort_values(["cell_id", "quarter"])
hist = df.copy()
hist["prev_burn"] = hist.groupby("cell_id")["burn_value"].shift(1).fillna(0)
hist["prev_fire"] = (hist["prev_burn"] > 0).astype(int)
hist["roll2_sum"] = hist.groupby("cell_id")["burn_value"].rolling(2, 1).sum().reset_index(level=0, drop=True)
hist["roll4_sum"] = hist.groupby("cell_id")["burn_value"].rolling(4, 1).sum().reset_index(level=0, drop=True)
hist_latest = hist.groupby("cell_id").tail(1).set_index("cell_id")

def fetch_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    r = requests.get(url, params={"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"}, timeout=10)
    j = r.json()
    precip = sum(float(v) for v in j.get("rain", {}).values()) if "rain" in j else 0
    return {"precipitation": precip, "temperature": j["main"]["temp"], "relative_humidity": j["main"]["humidity"],
            "wind_speed": j["wind"].get("speed", 0), "wind_direction": j["wind"].get("deg", 0)}

def fetch_forecast(lat, lon):
    """Fetch 5-day/3-hour forecast and return first 48 hours (16 entries)."""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    entries = r.json().get("list", [])[:16] 

    out = []
    for e in entries:
        main = e.get("main", {})
        wind = e.get("wind", {})
        rain = e.get("rain", {})
        out.append({
            "timestamp": e["dt"],
            "precipitation": float(rain.get("3h", 0.0)),
            "temperature": float(main.get("temp")),
            "relative_humidity": float(main.get("humidity")),
            "wind_speed": float(wind.get("speed", 0.0)),
            "wind_direction": float(wind.get("deg", 0.0)),
        })
    return out


def get_forecast_all():
    """Fetch 48-hour forecast for all 36 regions."""
    fcache = {}
    for _, r in gdf.iterrows():
        rid = int(r.name)
        lat, lon = r.geometry.centroid.y, r.geometry.centroid.x
        try:
            fcache[rid] = fetch_forecast(lat, lon)
        except:
            fcache[rid] = []
        time.sleep(0.15)
    return fcache

def get_weather():
    cache = {}
    for _, r in gdf.iterrows():
        try: cache[int(r.name)] = fetch_weather(r.geometry.centroid.y, r.geometry.centroid.x)
        except: cache[int(r.name)] = {"precipitation": 0}
        time.sleep(0.1)
    return cache

should_refresh = False
if ss.auto_refresh and (time.time() - ss.last_refresh >= REFRESH_INTERVAL): should_refresh = True
if ss.force_refresh: should_refresh = True; ss.force_refresh = False

if should_refresh or not ss.weather:
    with st.spinner("Fetching weather / forecast and predictions..."):

        if forecast_mode:
            ss.weather = get_forecast_all()
        else:
            ss.weather = get_weather()

        rec = []
        for _, r in gdf.iterrows():
            rid = int(r.name)
            if forecast_mode:
                fl = ss.weather.get(rid, [])
                idx = min(forecast_hour // 3, len(fl) - 1)
                w = fl[idx] if fl else {"precipitation":0,"temperature":0,"relative_humidity":0,"wind_speed":0,"wind_direction":0}
            else:
                w = ss.weather.get(rid, {})

            histrow = hist_latest.loc[rid] if rid in hist_latest.index else None
            t, h, ws, wd = w.get("temperature",0), w.get("relative_humidity",0), w.get("wind_speed",0), w.get("wind_direction",0)
            wx, wy = np.cos(np.deg2rad(wd)), np.sin(np.deg2rad(wd))

            rec.append({
                "region_id": rid,
                "precipitation": w.get("precipitation", 0),
                "temperature": t,
                "relative_humidity": h,
                "wind_speed": ws,
                "wind_x": wx,
                "wind_y": wy,
                "temp_wind": t*ws,
                "hum_temp_ratio": h/(t if t!=0 else 1e-9),
                "prev_burn": float(histrow["prev_burn"]) if histrow is not None else 0,
                "prev_fire": int(histrow["prev_fire"]) if histrow is not None else 0,
                "roll2_sum": float(histrow["roll2_sum"]) if histrow is not None else 0,
                "roll4_sum": float(histrow["roll4_sum"]) if histrow is not None else 0,
            })
        X = pd.DataFrame(rec).set_index("region_id")[FEATURES].fillna(0.0)
        probs = np.column_stack([predict_proba_safe(base_models[n], X) for n in model_names])
        if artifact_type == "weighted" and weights is not None:
            ens = probs.dot(weights)
        elif artifact_type == "stacking" and meta_model is not None and meta_scaler is not None:
            ens = meta_model.predict_proba(meta_scaler.transform(probs))[:, 1]
        else:
            ens = probs.mean(axis=1)
        ss.preds = {int(r): float(p) for r, p in zip(X.index, ens)}
        ss.last_refresh = time.time()
        ss.last_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ss.last_refresh))
        for rid, prob in ss.preds.items():
            if prob >= HIGH_THRESH:
                play_alarm(); send_windows_alert(rid, prob); send_email_alert(rid, prob)

if os.path.exists("images/wildfire_header.jpg"):
    header_image_path = "images/wildfire_header.jpg"
    with open(header_image_path, "rb") as f:
        header_base64 = base64.b64encode(f.read()).decode()
    header_img_tag = f"<img src='data:image/jpeg;base64,{header_base64}' style='width:280px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.2);'>"
else:
    header_img_tag = "<div style='width:280px;height:180px;background:#eee;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#999;'>No image</div>"

st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:center;gap:40px;margin-top:10px;margin-bottom:20px;flex-wrap:wrap;'>
    {header_img_tag}
    <div style='max-width:550px;text-align:left;'>
        <h1 style='color:#b33a3a;font-size:38px;margin-bottom:5px;'>🔥 Kelowna Wildfire Risk Forecast Portal</h1>
        <p style='font-size:18px;color:#444;margin-top:5px;margin-bottom:10px;'>
            Real-time wildfire predictions for <b>your community</b>.<br>
            Know the risk. Take action. Stay safe.
        </p>
        <p style='font-size:15px;color:#777;margin-top:5px;'>
            ⏱ Auto-updates every 5 minutes | 🌐 Data: OpenWeatherMap + BC Wildfire Service | 🤖 Powered by AI
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("""
### 🧠 What the AI Monitors
- 🌡 **Temperature** — Higher heat = drier vegetation  
- 💧 **Humidity** — Low humidity = faster fire spread  
- 🌬 **Wind** — Speed & direction drive fire behavior  
- 🌧 **Rainfall** — Reduces ignition likelihood  
- 🔥 **Fire History** — Past burns indicate vulnerability
""")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>🗺 Forecast Fire Risk Map</h2>", unsafe_allow_html=True)
if ss.preds:
    m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, tiles="CartoDB positron")
    for _, row in gdf.iterrows():
        rid = int(row.name)
        p = ss.preds.get(rid, 0)

        raw = ss.weather.get(rid, {})
        if isinstance(raw, list):
            if len(raw) == 0:
                w = {}
            else:
                idx = min(int(ss.get("forecast_hour", 0)) // 3, len(raw) - 1)
                w = raw[idx]
        else:
            w = raw
        p=p/2.0
        forecast_text = ""
        if ss.get("forecast_mode", False):
            fh = ss.get("forecast_hour", 0)
            forecast_text = f"<br>⏱️ <i>Forecast: {fh} hours ahead</i><br>"

        popup_html = f"""
        <b style='color:#b33a3a;'>Region {rid}</b><br><br>
        {forecast_text}
        🔥 Fire Risk: <b style='color:{"#c62828" if p>=HIGH_THRESH else "#ff6f00" if p>=0.5 else "#388e3c"}'>{p:.1%}</b><br>
        🌡 Temperature: {w.get('temperature','N/A')} °C<br>
        💧 Humidity: {w.get('relative_humidity','N/A')}%<br>
        🌬 Wind: {w.get('wind_speed','N/A')} m/s<br>
        🌧 Rain: {w.get('precipitation','N/A')} mm<br>
        """

        folium.GeoJson(
            data=json.loads(gpd.GeoSeries([row.geometry]).to_json())["features"][0],
            style_function=lambda x, p=p: {"fillColor": color_for_prob(p), "color": "#333", "weight": 1, "fillOpacity": 0.7},
            popup=Popup(popup_html, max_width=300)
        ).add_to(m)
    st.components.v1.html(m._repr_html_(), height=420)
else:
    st.warning("No predictions available.")

st.markdown(f"""
<div style='text-align:left; font-size:10px; margin-top:2px; margin-bottom:5px;'>
    <span style='font-size:10px;'>🕒 <b>Last updated:</b> {ss.last_time}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
ui_value = st.slider(
    "Forecast (hours ahead)",
    min_value=0,
    max_value=48,
    step=3,
    value=ss.get("forecast_hour", 0),
    key="forecast_ui",
    help="Choose how many hours ahead to forecast (0 = realtime)"
)
st.markdown("</div>", unsafe_allow_html=True)

if ui_value != ss["forecast_hour"]:
    ss["forecast_hour"] = ui_value
    ss["forecast_mode"] = (ui_value > 0)
    ss["force_refresh"] = True
    st.rerun()


left, middle, right = st.columns([0.5, 1, 0.3])

with middle:
    btn_col, chk_col = st.columns([1, 1.5], gap="medium")

    with btn_col:
        refresh_now = st.button("🔄 Refresh", key="refresh_small", use_container_width=True)

    with chk_col:
        auto_toggle = st.checkbox("Auto-refresh every 5 min", value=ss.auto_refresh)

if auto_toggle != ss.auto_refresh: ss.auto_refresh = auto_toggle; st.rerun()
if refresh_now: ss.force_refresh = True; st.rerun()

forecast_hour = int(ss.get("forecast_hour", 0))
forecast_mode = (forecast_hour > 0)


st.markdown("""
    <div class='legend-box'>
    <b>🎯 How to Read This Map:</b><br>
    <span style='color:#388e3c;'>●</span> <b>Green:</b> Low risk—normal precautions<br>
    <span style='color:#ffd600;'>●</span> <b>Yellow:</b> Moderate—stay aware<br>
    <span style='color:#ff6f00;'>●</span> <b>Orange:</b> Elevated—avoid open flames<br>
    <span style='color:#c62828;'>●</span> <b>Red:</b> High risk—be ready to evacuate
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

if os.path.exists("images/awareness.jpg"):
    with open("images/awareness.jpg", "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style='display:flex;justify-content:center;margin-top:25px;margin-bottom:15px;'>
            <img src='data:image/jpeg;base64,{img_data}' alt='Awareness' 
                 style='width:480px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.15);'>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
    f"""
    <div style='width:100%; text-align:center; margin-top:-8px;'>
        <span style='color:#ccc; font-size:14px;'>Stay alert. Prevent wildfires.</span>
    </div>
    """,
    unsafe_allow_html=True
    )


st.markdown("<h2 class='section-title'>🚨 Your Wildfire Action Plan</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <h4>🏠 Before Fire Season (Do This Now)</h4>
    <ul>
        <li>Create a <b>go-bag</b> with 3 days of supplies.</li>
        <li>Take photos of home & belongings for insurance.</li>
        <li>Clear gutters, rake leaves, trim branches.</li>
        <li>Know <b>two ways out</b> of your neighborhood.</li>
    </ul>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <h4>🔥 When Fire Threatens (Act Fast)</h4>
    <ul>
        <li>Stay calm, pack essentials, close all windows.</li>
        <li>Follow evacuation alerts promptly.</li>
        <li>If ordered to evacuate — <b>leave immediately!</b></li>
        <li>Take pets, turn off gas, inform someone of your route.</li>
    </ul>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <h4>🌧 After the Fire (Stay Safe)</h4>
    <ul>
        <li>Wait for the official all-clear before returning.</li>
        <li>Watch for damaged trees, power lines, hotspots.</li>
        <li>Document damage with photos before cleanup.</li>
    </ul>
    """, unsafe_allow_html=True)


if os.path.exists("images/evacuation_banner.jpg"):
    with open("images/evacuation_banner.jpg", "rb") as f:
        evac_data = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style='display:flex;justify-content:center;margin-top:25px;margin-bottom:15px;'>
            <img src='data:image/jpeg;base64,{evac_data}' alt='Evacuation Banner' 
                 style='width:480px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.15);'>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
    f"""
    <div style='width:100%; text-align:center; margin-top:-8px;'>
        <span style='color:#ccc; font-size:14px;'>Evacuation & Safety Guidelines</span>
    </div>
    """,
    unsafe_allow_html=True
    )


st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("""
<h2 class='section-title'>📞 Emergency Contacts</h2>
<div class='contact-card'>
<b>🔥 BC Wildfire Service:</b> 1-800-663-5555<br>
<b>🚓 Emergency Services:</b> 911<br>
<b>🏙 City of Kelowna Info:</b> 250-469-8500<br>
<b>📱 Report Smoke:</b> *#5555 from your cell*
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;color:#999;font-size:13px;margin-top:30px;'>Kelowna Wildfire Risk Portal © 2025 | AI-powered safety dashboard</div>", unsafe_allow_html=True)

