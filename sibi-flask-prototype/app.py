from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, abort, jsonify
from werkzeug.utils import secure_filename
import os
import csv

from datetime import datetime
from pathlib import Path
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp

BASE_DIR = Path(__file__).parent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-key'
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'static' / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

# Gambar peraga abjad SIBI untuk halaman Kamus — versi sudah di-crop rapat ke
# konten tangan (lihat static/images/abjad/), supaya semua kartu tampil konsisten.
ABJAD_DIR = BASE_DIR / 'static' / 'images' / 'abjad'

# Data landmark hasil konfirmasi user (real-time translate) disimpan di sini.
# Training tetap dilakukan manual/offline — file ini cuma tempat penampungan,
# tidak ada retraining otomatis dari sini.
COLLECT_DIR = BASE_DIR / 'collected_data'
COLLECT_FILE = COLLECT_DIR / 'samples.csv'

# ============================================================
# LOAD MODEL  (sibi_realtime_random_forest_bundle.joblib — dict
# {"model": RandomForestClassifier, "scaler": StandardScaler, "classes": [0..25], ...})
# ============================================================

_bundle_path = BASE_DIR / 'sibi_realtime_random_forest_bundle.joblib'

# Model baru output-nya angka (0=A, 1=B, ..., 25=Z), bukan huruf,
# jadi didekode manual supaya rule I/Y, U/R/V, O tetap bisa jalan.
ID_TO_LABEL = {i: chr(ord('A') + i) for i in range(26)}
ABJAD_LETTERS = [ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)]

try:
    _bundle     = joblib.load(_bundle_path)
    best_model  = _bundle['model']
    scaler      = _bundle['scaler']
    _model_type = _bundle.get('model_type', 'Unknown')
    _model_name = _bundle.get('model_name', 'Unknown')
    label_names = ABJAD_LETTERS
    MODEL_READY = True
    print(f"[SIBI] Model loaded: {_model_name} ({_model_type}) | {len(label_names)} classes")

except Exception as e:
    MODEL_READY = False
    _model_name = 'Unknown'
    print(f"[SIBI] Model tidak bisa di-load: {e}")


def decode_label(raw_label):
    try:
        return ID_TO_LABEL[int(raw_label)]
    except (KeyError, ValueError, TypeError):
        return str(raw_label)


# ============================================================
# CONFIG  (dari realtime_sibi_webcam.py)
# ============================================================

MAX_HANDS            = 2
LANDMARKS_PER_HAND   = 21
COORDS_PER_LANDMARK  = 3
FEATURE_DIM          = 63  # single-hand: 21 × 3

CONF_THRESHOLD                   = 0.15
SMOOTH_WINDOW                    = 8
MAX_SAMPLE_FRAMES                = 120
MIN_DETECTION_CONFIDENCE         = 0.30
MIN_TRACKING_CONFIDENCE          = 0.30

ENABLE_IY_RULE                   = True
ENABLE_UV_RULE                   = True
ENABLE_O_RULE                    = True

LOW_CONF_RULE_OVERRIDE_THRESHOLD = 0.25
O_THUMB_INDEX_THRESHOLD          = 0.30
O_MAX_TOP1_CONF_FOR_OVERRIDE     = 0.45


# ============================================================
# FEATURE EXTRACTION  (single-hand, 63 fitur)
# ============================================================

def hand_area(hand_landmarks):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def select_best_hand(multi_hand_landmarks):
    """Ambil tangan dengan area terbesar."""
    if not multi_hand_landmarks:
        return None
    if len(multi_hand_landmarks) == 1:
        return multi_hand_landmarks[0]
    areas = [hand_area(h) for h in multi_hand_landmarks]
    return multi_hand_landmarks[int(np.argmax(areas))]


def get_hand_coords(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)


def normalize_single_hand_landmarks(hand_coords):
    """Input (21,3) → output (1,63)."""
    hand_coords = hand_coords.astype(np.float32)
    wrist    = hand_coords[0].copy()
    centered = hand_coords - wrist
    scale    = np.max(np.linalg.norm(centered[:, :2], axis=1))
    if scale < 1e-6:
        xy_range = np.max(hand_coords[:, :2], axis=0) - np.min(hand_coords[:, :2], axis=0)
        scale = np.max(xy_range)
    if scale < 1e-6:
        raise ValueError("invalid_landmark_scale")
    feature = (centered / scale).reshape(-1).astype(np.float32)
    if feature.shape[0] != FEATURE_DIM:
        raise ValueError(f"Feature harus {FEATURE_DIM}, dapat {feature.shape[0]}")
    return feature.reshape(1, -1)


def build_feature_from_result(multi_hand_landmarks):
    """Return (features, selected_hand_landmarks, detected_hands)."""
    if not multi_hand_landmarks:
        return None, None, 0
    detected_hands = len(multi_hand_landmarks)
    selected = select_best_hand(multi_hand_landmarks)
    if selected is None:
        return None, None, detected_hands
    coords   = get_hand_coords(selected)
    features = normalize_single_hand_landmarks(coords)
    return features, selected, detected_hands


def predict_label(features):
    """Return (label, confidence, top3, topk)."""
    features_scaled = scaler.transform(features).astype(np.float32)

    if hasattr(best_model, 'predict_proba'):
        proba    = best_model.predict_proba(features_scaled)[0]
        top_cols = np.argsort(proba)[::-1][:5]
        topk = []
        for col_idx in top_cols:
            class_id = int(best_model.classes_[col_idx])
            label    = decode_label(class_id)
            topk.append((label, float(proba[col_idx])))
        return topk[0][0], topk[0][1], topk[:3], topk

    pred_id = int(best_model.predict(features_scaled)[0])
    label   = decode_label(pred_id)
    return label, 1.0, [(label, 1.0)], [(label, 1.0)]


# ============================================================
# GEOMETRY
# ============================================================

def _euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)
    d = np.array(d[:2], dtype=np.float32)
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


# ============================================================
# FINGER STATE
# ============================================================

def estimate_finger_states(hand_coords):
    if hand_coords is None or hand_coords.shape[0] < 21:
        return None

    wrist       = hand_coords[0]
    index_mcp   = hand_coords[5]
    middle_mcp  = hand_coords[9]
    pinky_mcp   = hand_coords[17]
    palm_center = (wrist + index_mcp + middle_mcp + pinky_mcp) / 4.0
    hand_scale  = _euclidean(wrist[:2], middle_mcp[:2])
    if hand_scale < 1e-6:
        hand_scale = 1.0

    states = {}
    for name, (tip, pip) in [('index',(8,6)), ('middle',(12,10)), ('ring',(16,14)), ('pinky',(20,18))]:
        tip_dist = _euclidean(hand_coords[tip][:2], palm_center[:2]) / hand_scale
        pip_dist = _euclidean(hand_coords[pip][:2], palm_center[:2]) / hand_scale
        states[name] = bool((tip_dist > pip_dist + 0.08) and (tip_dist > 0.75))

    thumb_tip         = hand_coords[4]
    thumb_ip          = hand_coords[3]
    thumb_tip_dist    = _euclidean(thumb_tip[:2], palm_center[:2]) / hand_scale
    thumb_ip_dist     = _euclidean(thumb_ip[:2],  palm_center[:2]) / hand_scale
    thumb_to_idx_mcp  = _euclidean(thumb_tip[:2], index_mcp[:2])   / hand_scale
    states['thumb'] = bool(
        (thumb_tip_dist > thumb_ip_dist + 0.05 and thumb_tip_dist > 0.65)
        or (thumb_to_idx_mcp > 0.95)
    )
    return states


# ============================================================
# RULES  (dari cobalagilagi.py)
# ============================================================

def rule_based_label_for_I_Y(hand_coords):
    states = estimate_finger_states(hand_coords)
    if states is None:
        return None, None
    t, i, m, r, p = states['thumb'], states['index'], states['middle'], states['ring'], states['pinky']
    if t and p and not i and not m and not r:
        return 'Y', states
    if not t and p and not i and not m and not r:
        return 'I', states
    return None, states


def rule_based_label_for_U_R_V(hand_coords):
    states = estimate_finger_states(hand_coords)
    if states is None:
        return None, None, {}
    if hand_coords is None or hand_coords.shape[0] < 21:
        return None, states, {}

    if not (states['index'] and states['middle'] and not states['ring'] and not states['pinky']):
        return None, states, {}

    wrist      = hand_coords[0]
    middle_mcp = hand_coords[9]
    hand_scale = _euclidean(wrist[:2], middle_mcp[:2])
    if hand_scale < 1e-6:
        hand_scale = 1.0

    index_mcp  = hand_coords[5]
    index_tip  = hand_coords[8]
    middle_mcp2 = hand_coords[9]
    middle_tip = hand_coords[12]

    tip_distance = _euclidean(index_tip[:2], middle_tip[:2]) / hand_scale
    is_crossed   = _segments_intersect(index_mcp, index_tip, middle_mcp2, middle_tip)
    debug_info   = {'tip_distance': tip_distance, 'is_crossed': is_crossed}

    if is_crossed:
        return 'R', states, debug_info
    if tip_distance < 0.50:
        return 'U', states, debug_info
    return 'V', states, debug_info


def rule_based_label_for_O(hand_coords):
    states = estimate_finger_states(hand_coords)
    if states is None:
        return None, None, {}
    if hand_coords is None or hand_coords.shape[0] < 21:
        return None, states, {}

    wrist      = hand_coords[0]
    middle_mcp = hand_coords[9]
    hand_scale = _euclidean(wrist[:2], middle_mcp[:2])
    if hand_scale < 1e-6:
        hand_scale = 1.0

    thumb_tip   = hand_coords[4]
    index_tip   = hand_coords[8]
    index_pip   = hand_coords[6]
    index_mcp   = hand_coords[5]
    middle_tip  = hand_coords[12]
    ring_tip    = hand_coords[16]
    pinky_tip   = hand_coords[20]

    thumb_index_dist  = _euclidean(thumb_tip[:2], index_tip[:2]) / hand_scale
    index_tip_to_mcp  = _euclidean(index_tip[:2], index_mcp[:2]) / hand_scale
    index_pip_to_mcp  = _euclidean(index_pip[:2], index_mcp[:2]) / hand_scale

    debug_info = {
        'thumb_index_dist': thumb_index_dist,
        'index_tip_to_mcp': index_tip_to_mcp,
        'index_pip_to_mcp': index_pip_to_mcp,
    }

    thumb_index_close   = thumb_index_dist < O_THUMB_INDEX_THRESHOLD
    index_not_straight  = not states['index']
    other_closed        = not states['middle'] and not states['ring'] and not states['pinky']

    if thumb_index_close and index_not_straight and other_closed:
        return 'O', states, debug_info
    return None, states, debug_info


def apply_rule_corrections(label, top3, hand_coords):
    """Terapkan semua rule koreksi secara bertahap. Return (corrected_label, states, used_rule, debug_info)."""
    top_labels   = [x[0] for x in top3]
    top2_labels  = top_labels[:2]
    top1_label   = top3[0][0]
    top1_conf    = top3[0][1]

    states         = None
    used_rule      = False
    debug_info     = {}
    corrected      = label

    # Rule I/Y
    if ENABLE_IY_RULE:
        rl, st = rule_based_label_for_I_Y(hand_coords)
        if st is not None:
            states = st
        can = top1_label in {'I', 'Y'} or (
            ('I' in top2_labels or 'Y' in top2_labels) and top1_conf < LOW_CONF_RULE_OVERRIDE_THRESHOLD
        )
        if rl and can:
            corrected = rl
            used_rule = True
            debug_info['rule'] = 'I/Y'

    # Rule U/R/V
    if ENABLE_UV_RULE:
        rl, st, dbg = rule_based_label_for_U_R_V(hand_coords)
        if st is not None:
            states = st
        can = top1_label in {'U', 'R', 'V'} or (
            ('U' in top2_labels or 'R' in top2_labels or 'V' in top2_labels)
            and top1_conf < LOW_CONF_RULE_OVERRIDE_THRESHOLD
        )
        if rl and can:
            corrected = rl
            used_rule = True
            debug_info['rule'] = 'U/R/V'
        if dbg:
            debug_info.update(dbg)

    # Rule O (konservatif)
    if ENABLE_O_RULE:
        rl, st, dbg = rule_based_label_for_O(hand_coords)
        if st is not None:
            states = st
        can = top1_label == 'O' or ('O' in top2_labels and top1_conf < O_MAX_TOP1_CONF_FOR_OVERRIDE)
        if rl and can:
            corrected = 'O'
            used_rule = True
            debug_info['rule'] = 'O'
        if dbg:
            debug_info.update(dbg)

    return corrected, states, used_rule, debug_info


# ============================================================
# RUN PREDICTION  (pipeline realtime_sibi_webcam.py — single-hand)
# ============================================================

def run_prediction(result):
    """Return (label, confidence, top3, used_rule, features)."""
    if not result.multi_hand_landmarks:
        return None, 0.0, [], False, None

    try:
        features, selected_hand, _ = build_feature_from_result(result.multi_hand_landmarks)
    except Exception:
        return None, 0.0, [], False, None

    if features is None or selected_hand is None:
        return None, 0.0, [], False, None

    label, conf, top3, _topk = predict_label(features)
    hand_coords               = get_hand_coords(selected_hand)
    corrected, _, used_rule, _ = apply_rule_corrections(label, top3, hand_coords)

    return corrected, conf, top3, used_rule, features


# ============================================================
# VIDEO PROCESSING
# ============================================================


def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV tidak bisa membuka video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[SIBI] total_frames={total_frames}, path={video_path}")
    step = max(1, total_frames // MAX_SAMPLE_FRAMES)

    mp_hands = mp.solutions.hands
    pred_buf  = deque(maxlen=SMOOTH_WINDOW)
    confidences   = []
    label_sequence = []
    prev_smoothed  = None
    frame_idx = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as detector:

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            label, conf, _, _, _ = run_prediction(result)

            if label and conf >= CONF_THRESHOLD:
                pred_buf.append(label)
                confidences.append(conf)
                smoothed = Counter(pred_buf).most_common(1)[0][0]
                if smoothed != prev_smoothed:
                    label_sequence.append(smoothed)
                    prev_smoothed = smoothed

            frame_idx += step
            if frame_idx >= total_frames:
                break

    cap.release()

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return label_sequence, avg_conf, len(confidences)


# ============================================================
# HISTORY
# ============================================================

HISTORY = [
    {
        'filename': 'demo_sibi_perkenalan.mp4',
        'date': '03 Jun 2026',
        'confidence': '92%',
        'translation': 'Halo, nama saya Hanny. Senang bertemu dengan kamu.',
        'status': 'Selesai',
    },
    {
        'filename': 'demo_sibi_kelas.mp4',
        'date': '02 Jun 2026',
        'confidence': '88%',
        'translation': 'Saya ingin belajar bersama di kelas hari ini.',
        'status': 'Selesai',
    },
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# ROUTES
# ============================================================

@app.route('/status')
def status():
    from flask import jsonify
    info = {
        'model_ready': MODEL_READY,
        'model_key': _model_name if MODEL_READY else None,
        'num_classes': len(label_names) if MODEL_READY else 0,
        'classes': label_names if MODEL_READY else [],
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.isdir(app.config['UPLOAD_FOLDER']),
    }
    return jsonify(info)


@app.route('/')
def index():
    stats = {
        'total_videos': len(HISTORY),
        'avg_confidence': '90%',
        'supported_format': 'MP4, MOV, AVI, MKV, WEBM',
        'model_status': 'Aktif' if MODEL_READY else 'Prototype Mode',
    }
    return render_template('index.html', history=HISTORY, stats=stats)


@app.route('/translate', methods=['GET', 'POST'])
def translate():
    result = None
    video_url = None

    if request.method == 'POST':
        file = request.files.get('video')

        if not file or file.filename == '':
            flash('Pilih file video terlebih dahulu ya.', 'error')
            return redirect(url_for('translate'))

        if not allowed_file(file.filename):
            flash('Format file belum didukung. Gunakan MP4, MOV, AVI, MKV, atau WEBM.', 'error')
            return redirect(url_for('translate'))

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        saved_filename = f'{timestamp}_{filename}'
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(save_path)
        video_url = url_for('static', filename=f'uploads/{saved_filename}')

        if MODEL_READY:
            try:
                labels, avg_conf, detected_frames = process_video(save_path)

                print(f"[SIBI] labels={labels}, conf={avg_conf:.2f}, frames={detected_frames}")
                if labels:
                    translation = ' '.join(labels)
                    confidence_str = f'{avg_conf * 100:.0f}%'
                    status = 'Selesai'
                    flash(f'Deteksi selesai — {detected_frames} frame, {len(labels)} isyarat: {translation}', 'success')
                else:
                    translation = '(Tidak ada tangan terdeteksi dalam video)'
                    confidence_str = '0%'
                    status = 'Tidak ada isyarat'
                    flash('Tangan tidak terdeteksi. Pastikan tangan terlihat jelas di video.', 'warning')

            except Exception as e:
                translation = f'(Error saat memproses: {e})'
                confidence_str = '0%'
                status = 'Error'
                flash(f'Terjadi kesalahan: {e}', 'error')
        else:
            translation = '(Model tidak tersedia — prototype mode)'
            confidence_str = '—'
            status = 'Prototype'
            flash('Model ML tidak bisa di-load. Berjalan dalam mode prototype.', 'warning')

        result = {
            'filename': filename,
            'date': datetime.now().strftime('%d %b %Y'),
            'confidence': confidence_str,
            'translation': translation,
            'status': status,
        }
        HISTORY.insert(0, result)

    return render_template('translate.html', result=result, video_url=video_url)


@app.route('/api/predict-frame', methods=['POST'])
def predict_frame():
    """Terima 1 frame webcam (base64 JPEG), return prediksi + top3."""
    from flask import jsonify
    import base64 as b64

    if not MODEL_READY:
        return jsonify({'error': 'Model tidak tersedia'}), 503

    data = request.get_json(silent=True) or {}
    image_b64 = data.get('image', '')
    if ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]

    try:
        img_bytes = b64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'error': 'Gambar tidak valid'}), 400

    if frame is None:
        return jsonify({'error': 'Gambar tidak valid'}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as detector:
        result = detector.process(rgb)

    label, confidence, top3, used_rule, features = run_prediction(result)
    features_list = features.flatten().tolist() if features is not None else None

    if not label:
        return jsonify({'label': None, 'confidence': 0, 'message': 'Tangan tidak terdeteksi'})

    if confidence >= CONF_THRESHOLD:
        return jsonify({
            'label': label,
            'confidence': round(float(confidence), 2),
            'top3': [{'label': l, 'conf': round(c, 2)} for l, c in top3],
            'rule': used_rule,
            'features': features_list,
        })
    return jsonify({
        'label': None,
        'confidence': round(float(confidence), 2),
        'message': 'Confidence rendah',
        'top3': [{'label': l, 'conf': round(c, 2)} for l, c in top3],
        'features': features_list,
    })


@app.route('/api/save-sample', methods=['POST'])
def save_sample():
    """Simpan 1 sample (fitur landmark + label terkonfirmasi user) ke CSV.
    Murni penyimpanan — training tetap dilakukan manual/offline dari file ini."""
    data = request.get_json(silent=True) or {}
    features = data.get('features')
    label = str(data.get('label', '')).strip().upper()
    predicted_label = data.get('predicted_label')
    confidence = data.get('confidence')

    if not isinstance(features, list) or len(features) != FEATURE_DIM:
        return jsonify({'error': f'features harus berupa list {FEATURE_DIM} angka'}), 400
    if not all(isinstance(x, (int, float)) for x in features):
        return jsonify({'error': 'features harus berisi angka'}), 400
    if label not in ABJAD_LETTERS:
        return jsonify({'error': 'label tidak valid'}), 400

    COLLECT_DIR.mkdir(parents=True, exist_ok=True)
    is_new_file = not COLLECT_FILE.exists()

    with open(COLLECT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow([f'feat_{i}' for i in range(FEATURE_DIM)] + ['label', 'predicted_label', 'confidence', 'timestamp'])
        writer.writerow(
            list(features)
            + [label, predicted_label or '', confidence if confidence is not None else '', datetime.now().isoformat(timespec='seconds')]
        )

    return jsonify({'status': 'ok'})


@app.route('/api/predict-video', methods=['POST'])
def predict_video_api():
    """Terima video blob (multipart), return prediksi sebagai JSON."""
    from flask import jsonify
    import tempfile

    if not MODEL_READY:
        return jsonify({'error': 'Model tidak tersedia'}), 503

    file = request.files.get('video')
    if not file:
        return jsonify({'error': 'Tidak ada file'}), 400

    suffix = '.' + (file.filename.rsplit('.', 1)[-1] if '.' in file.filename else 'webm')
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        labels, avg_conf, frames = process_video(tmp_path)
    finally:
        os.unlink(tmp_path)

    if labels:
        return jsonify({
            'labels': labels,
            'translation': ' '.join(labels),
            'confidence': round(avg_conf * 100),
            'frames': frames,
        })
    return jsonify({'labels': [], 'translation': '', 'confidence': 0, 'frames': frames,
                    'message': 'Tangan tidak terdeteksi'})


@app.route('/history')
def history():
    return render_template('history.html', letters=ABJAD_LETTERS)


@app.route('/abjad/<letter>.png')
def abjad_image(letter):
    letter = letter.upper()
    if letter not in ABJAD_LETTERS:
        abort(404)
    return send_from_directory(ABJAD_DIR, f'{letter}.png')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/edukasi')
def edukasi():
    return render_template('edukasi.html')


if __name__ == '__main__':
    # ssl_context='adhoc' bikin sertifikat self-signed on-the-fly, supaya
    # browser (termasuk di HP) menganggap ini "secure context" — wajib
    # buat izin akses kamera (getUserMedia) di luar localhost. Browser akan
    # tetap tampilkan warning "Not Secure/Your connection isn't private",
    # itu normal untuk self-signed cert — klik "Advanced" > "Proceed anyway".
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')
