# --- Auto-install ---
import importlib.util, subprocess, sys

def ensure_package(pkg):
    if importlib.util.find_spec(pkg) is None:
        print(f"Installerer {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in ["pillow", "pandas", "requests", "ultralytics"]:
    ensure_package(p)




# keydetector.py
import os, requests
from collections import defaultdict, deque
from PIL import Image, ImageDraw, ImageFont
import pandas as pd


# -------------------- DEFAULT SETTINGS --------------------
API_KEY_DEFAULT  = "iiTrZdELA9a8xnb763Fa"              #    Insert API key here
MODEL_ID_DEFAULT = "keyboard-key-recognition-kw7nc/14" #    Insert model here
CONF_DEFAULT = 0.15
ALLOW_REUSE_DEFAULT = True

CHAR2CLASS = {
    " ": "space", ",": "comma", ".": "point", "-": "minus", "+": "plus",
    "<": "less", "^": "caret", "#": "hash", "\t": "tab", "\n": "enter",
    "æ": "ae", "ø": "oe", "ü": "ue",
}


# -------------------- HELPER FUNCTIONS --------------------
def char_to_class(ch: str):
    if ch.isalpha(): return ch.lower()
    if ch.isdigit(): return ch
    return CHAR2CLASS.get(ch)

def safe_font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        l,t,r,b = draw.textbbox((0,0), text, font=font)
        return (r-l, b-t)
    return draw.textsize(text, font=font)


# -------------------- MAIN FUNCTION --------------------
def detect_keyboard_text(
    image_path,
    target_text,
    out_dir,
    api_key=API_KEY_DEFAULT,
    model_id=MODEL_ID_DEFAULT,
    conf=CONF_DEFAULT,
    allow_reuse=ALLOW_REUSE_DEFAULT,
    save_visuals=True
):
    """
    Detect keys from a keyboard image using a Roboflow model.

    Returns:
        df (pd.DataFrame): results table
        centers (list of tuples): [(cx, cy), ...]
        chars (list of str): detected character sequence
    """

    os.makedirs(out_dir, exist_ok=True)
    out_sub = os.path.join(out_dir, target_text.replace(" ", "_"))
    os.makedirs(out_sub, exist_ok=True)

    # ---------- 1) Call Roboflow API ----------
    url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}&confidence={conf}&format=json"
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"file": f})
    resp.raise_for_status()
    rf_result = resp.json()

    # ---------- 2) Convert detections ----------
    img = Image.open(image_path).convert("RGB")
    detections = []
    for p in rf_result.get("predictions", []):
        cname = p["class"]
        if cname.startswith("key_"):
            cname = cname[4:]
        cname = cname.lower()

        x1 = int(p["x"] - p["width"]/2)
        y1 = int(p["y"] - p["height"]/2)
        x2 = int(p["x"] + p["width"]/2)
        y2 = int(p["y"] + p["height"]/2)

        detections.append({
            "class": cname,
            "conf": float(p["confidence"]),
            "xyxy": [x1,y1,x2,y2],
            "cx": (x1+x2)/2.0,
            "cy": (y1+y2)/2.0,
            "w": x2-x1,
            "h": y2-y1
        })

    # ---------- 3) Match detections to text ----------
    by_class = defaultdict(list)
    for d in detections:
        by_class[d["class"]].append(d)
    for k in by_class:
        by_class[k].sort(key=lambda d: (d["conf"], d["w"]*d["h"]), reverse=True)
        by_class[k] = deque(by_class[k])

    matched = []
    for ch in target_text:
        wanted = char_to_class(ch)
        det = None
        if wanted in by_class and len(by_class[wanted]) > 0:
            det = (by_class[wanted][0] if allow_reuse else by_class[wanted].popleft())
        matched.append((ch, wanted, det))

    # ---------- 4) Create panels + dataframe ----------
    font = safe_font(18)
    font_ov = safe_font(20)
    rows = []

    for idx, (ch, wanted, det) in enumerate(matched, start=1):
        panel_w, panel_h = 300, 120
        crop = None
        if det is not None:
            x1, y1, x2, y2 = det["xyxy"]
            crop = img.crop((x1, y1, x2, y2))
            panel_w = crop.width + 300
            panel_h = max(120, crop.height)

        panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
        draw_panel = ImageDraw.Draw(panel)
        if crop is not None:
            panel.paste(crop, (0, 0))

        tx = (0 if crop is None else crop.width) + 12
        ty = 12
        draw_panel.text((tx, ty), f"text char : {repr(ch)}", fill=(0,0,0), font=font); ty += 24
        draw_panel.text((tx, ty), f"wanted    : {wanted}", fill=(0,0,0), font=font); ty += 24

        row = {
            "pos": idx, "char": ch, "wanted": wanted,
            "found": None, "conf": None, "x1": None, "y1": None, "x2": None, "y2": None,
            "w": None, "h": None, "cx": None, "cy": None, "panel_path": None
        }

        if det is None:
            draw_panel.text((tx, ty), "FOUND     : (none)", fill=(200,0,0), font=font)
        else:
            draw_panel.text((tx, ty), f"FOUND     : {det['class']}", fill=(0,120,0), font=font); ty += 24
            draw_panel.text((tx, ty), f"conf      : {det['conf']:.3f}", fill=(0,0,0), font=font); ty += 24
            draw_panel.text((tx, ty), f"xyxy      : {int(det['xyxy'][0])},{int(det['xyxy'][1])},{int(det['xyxy'][2])},{int(det['xyxy'][3])}", fill=(0,0,0), font=font); ty += 24
            draw_panel.text((tx, ty), f"size      : {int(det['w'])}x{int(det['h'])} px", fill=(0,0,0), font=font)

            row.update({
                "found": det["class"], "conf": round(det["conf"], 6),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "w": int(det["w"]), "h": int(det["h"]),
                "cx": round(det["cx"], 2), "cy": round(det["cy"], 2)
            })

        # Save per-character panel
        panel_path = os.path.join(out_sub, f"{idx:03d}_{wanted or 'unknown'}.png")
        panel.save(panel_path)
        row["panel_path"] = panel_path
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------- 5) Overlays ----------
    if save_visuals:
        # === Overlay for alle detections ===
        overlay_all = img.copy()
        draw_all = ImageDraw.Draw(overlay_all)
        for det in detections:
            x1,y1,x2,y2 = det["xyxy"]
            cn, sc = det["class"], det["conf"]
            draw_all.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)
            label = f"{cn} {sc:.2f}"
            tw, th = text_size(draw_all, label, font_ov)
            draw_all.rectangle([x1, y1-th-6, x1+tw+8, y1], fill=(0,255,0))
            draw_all.text((x1+4, y1-th-4), label, fill=(0,0,0), font=font_ov)
        overlay_all.save(os.path.join(out_sub, "overlay_full.png"))

        # === Overlay for mached letters ===
        overlay_matched = img.copy()
        draw_m = ImageDraw.Draw(overlay_matched)
        for _, r in df.iterrows():
            if pd.isna(r["x1"]): 
                continue
            x1,y1,x2,y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            draw_m.rectangle([x1,y1,x2,y2], outline=(0,0,255), width=3)
            label = f"{r['wanted']} ({r['conf']:.2f})"
            tw, th = text_size(draw_m, label, font_ov)
            draw_m.rectangle([x1, y1-th-6, x1+tw+8, y1], fill=(0,0,255))
            draw_m.text((x1+4, y1-th-4), label, fill=(255,255,255), font=font_ov)
        overlay_matched.save(os.path.join(out_sub, "overlay_matched.png"))


    # ---------- 6) Extract centers ----------
    ordered = df.sort_values("pos")
    found_mask = ordered["found"].notna()
    ordered_found = ordered[found_mask].reset_index(drop=True)
    centers = [(float(r.cx), float(r.cy)) for _, r in ordered_found.iterrows()]
    chars = ordered_found["wanted"].tolist()

    return df, centers, chars


def detect_ghj(
    image_path,
    out_dir):

    _, centers, _ = detect_keyboard_text(image_path=image_path, target_text="ghj", out_dir=out_dir)
    return centers
