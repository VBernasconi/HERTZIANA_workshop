#!/usr/bin/env python3
"""
pose_gallery.py: match a live webcam pose to poses found in a folder of artworks.

SETUP (once)
    pip install "gradio>=5,<6" ultralytics opencv-python pillow numpy
    pip install controlnet-aux          # only needed for --backend openpose

USAGE
    # 1) Preprocess the artworks (slow, run once, or again when the folder changes)
    python pose_gallery.py preprocess ./artworks --cache ./cache --backend openpose

    # 2) Start the web app, then open http://127.0.0.1:7860 and allow the webcam
    python pose_gallery.py run --cache ./cache --backend yolo

Preprocessing stores, for every person found in every artwork, the keypoints and an image with that
person's skeleton drawn on top. The live app only compares joint angles + bone unit vectors, so
preprocessing and live detection may use different backends (e.g. OpenPose offline, YOLO live).
"""
import argparse, sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ------------------------------------------------------------------ settings
MIN_CONF = 0.3             # ignore keypoints below this confidence
MIN_KEYPOINTS = 6          # ignore detections with fewer visible joints
MAX_PEOPLE = 10            # per artwork: keep the largest N people
MAX_SIDE = 1024            # artworks are downscaled to this size (detection and display)
W_ANGLE, W_VEC = 0.5, 0.5  # weight of joint angles vs bone directions
MIN_SHARED_FRAC = 0.75     # compare a feature group only if >=75% of it is visible in BOTH poses
MIRROR_INVARIANT = True    # a left/right mirrored pose counts as the same pose
HYSTERESIS_DEG = 2.0       # live: keep the current artwork unless a new one is better by this margin
STREAM_EVERY = 0.5         # live: seconds between webcam frames sent to the model

KP_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
            "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
IDX = {n: i for i, n in enumerate(KP_NAMES)}

ANGLES = {   # angle at the MIDDLE joint
    "r_elbow": ("r_shoulder", "r_elbow", "r_wrist"), "l_elbow": ("l_shoulder", "l_elbow", "l_wrist"),
    "r_shoulder": ("neck", "r_shoulder", "r_elbow"), "l_shoulder": ("neck", "l_shoulder", "l_elbow"),
    "r_hip": ("neck", "r_hip", "r_knee"),            "l_hip": ("neck", "l_hip", "l_knee"),
    "r_knee": ("r_hip", "r_knee", "r_ankle"),        "l_knee": ("l_hip", "l_knee", "l_ankle"),
}
BONES = {    # unit vector from first to second point
    "neck_nose": ("neck", "nose"), "neck_r_shoulder": ("neck", "r_shoulder"), "neck_l_shoulder": ("neck", "l_shoulder"),
    "r_upper_arm": ("r_shoulder", "r_elbow"), "r_forearm": ("r_elbow", "r_wrist"),
    "l_upper_arm": ("l_shoulder", "l_elbow"), "l_forearm": ("l_elbow", "l_wrist"),
    "r_torso": ("neck", "r_hip"), "l_torso": ("neck", "l_hip"),
    "r_thigh": ("r_hip", "r_knee"), "r_shin": ("r_knee", "r_ankle"),
    "l_thigh": ("l_hip", "l_knee"), "l_shin": ("l_knee", "l_ankle"),
}
FACE_LINKS = [("nose", "r_eye"), ("nose", "l_eye"), ("r_eye", "r_ear"), ("l_eye", "l_ear")]   # drawing only


# ------------------------------------------------------------------ keypoint backends
COCO_TO_OP = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

def coco17_to_openpose18(p):
    out = np.full((18, 2), np.nan)
    for i, j in enumerate(COCO_TO_OP):
        if j is not None:
            out[i] = p[j]
    out[1] = (p[5] + p[6]) / 2          # neck = midpoint of the shoulders
    return out

_BACKENDS = {}

def get_backend(name):
    """Returns detect(img_rgb) -> list of (18, 2) float arrays (pixel coords, NaN = missing)."""
    if name in _BACKENDS:
        return _BACKENDS[name]

    if name == "yolo":
        from ultralytics import YOLO
        model = YOLO("yolo11n-pose.pt")             # use yolo11m-pose.pt for better accuracy (slower)

        def detect(img_rgb):
            r = model(np.ascontiguousarray(img_rgb[:, :, ::-1]), verbose=False)[0]   # ultralytics expects BGR
            if r.keypoints is None or len(r.keypoints) == 0:
                return []
            xy = r.keypoints.xy.cpu().numpy()
            cf = r.keypoints.conf
            cf = cf.cpu().numpy() if cf is not None else np.ones(xy.shape[:2])
            people = []
            for pts, c in zip(xy, cf):
                pts = pts.astype(float).copy()
                pts[(c < MIN_CONF) | (pts == 0).all(1)] = np.nan
                people.append(coco17_to_openpose18(pts))
            return people

    elif name == "mediapipe":
        detect = get_mediapipe_backend()
        
    elif name == "openpose":
        _mp = sys.modules.get("mediapipe")          # controlnet-aux imports a removed MediaPipe API;
        sys.modules["mediapipe"] = None             # OpenPose doesn't need it, so hide it for the import
        try:
            from controlnet_aux import OpenposeDetector
        finally:
            if _mp is not None: sys.modules["mediapipe"] = _mp
            else: sys.modules.pop("mediapipe", None)
        op = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        try:
            import torch
            if torch.cuda.is_available():
                op.to("cuda")
        except Exception:
            pass

        def detect(img_rgb):
            H, W = img_rgb.shape[:2]
            people = []
            for p in op.detect_poses(img_rgb):      # needs controlnet-aux >= 0.0.7
                if p.body is None:
                    continue
                kps = np.full((18, 2), np.nan)
                for i, k in enumerate(p.body.keypoints[:18]):
                    if k is not None and k.score >= MIN_CONF:
                        kps[i] = [k.x * W, k.y * H]
                people.append(kps)
            return people
    else:
        raise ValueError(f"unknown backend: {name}")

    _BACKENDS[name] = detect
    return detect

def get_mediapipe_backend():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = "pose_landmarker_lite.task"

    if not Path(model_path).exists():
        raise FileNotFoundError(
            "Download pose_landmarker_lite.task from MediaPipe "
            "and put it next to pose_gallery.py"
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.PoseLandmarker.create_from_options(
        options
    )

    # MediaPipe landmark indices to your OpenPose-18 format
    MP_MAP = {
        0: 0,    # nose
        12: 2,   # right shoulder
        14: 3,   # right elbow
        16: 4,   # right wrist
        11: 5,   # left shoulder
        13: 6,   # left elbow
        15: 7,   # left wrist
        24: 8,   # right hip
        26: 9,   # right knee
        28: 10,  # right ankle
        23: 11,  # left hip
        25: 12,  # left knee
        27: 13,  # left ankle
        5: 14,   # right eye
        2: 15,   # left eye
        8: 16,   # right ear
        7: 17    # left ear
    }

    frame_timestamp = [0]

    def detect(img_rgb):
        h, w = img_rgb.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(img_rgb)
        )

        frame_timestamp[0] += 33

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp[0]
        )

        people = []

        for pose in result.pose_landmarks:
            kps = np.full((18, 2), np.nan)

            for mp_idx, op_idx in MP_MAP.items():
                p = pose[mp_idx]

                if p.visibility >= MIN_CONF:
                    kps[op_idx] = [
                        p.x * w,
                        p.y * h
                    ]

            # Neck = midpoint between shoulders
            if not np.isnan(kps[2]).any() and not np.isnan(kps[5]).any():
                kps[1] = (kps[2] + kps[5]) / 2

            people.append(kps)

        return people

    return detect


def detect_people(img_rgb, backend):
    """People in an image, largest first, weak detections removed."""
    scored = []
    for kps in get_backend(backend)(img_rgb):
        ok = ~np.isnan(kps[:, 0])
        if ok.sum() >= MIN_KEYPOINTS:
            scored.append((np.ptp(kps[ok, 0]) * np.ptp(kps[ok, 1]), kps))
    scored.sort(key=lambda t: -t[0])
    return [k for _, k in scored[:MAX_PEOPLE]]


# ------------------------------------------------------------------ features and matching
def angle_at(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if not (n1 > 1e-6 and n2 > 1e-6):               # also False for NaN
        return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.full(2, np.nan)

def pose_features(kps):
    ang = np.array([angle_at(kps[IDX[a]], kps[IDX[b]], kps[IDX[c]]) for a, b, c in ANGLES.values()])
    vec = np.array([unit(kps[IDX[q]] - kps[IDX[p]]) for p, q in BONES.values()])
    return ang, vec

def _flip(n):
    return "l_" + n[2:] if n.startswith("r_") else "r_" + n[2:] if n.startswith("l_") else n
_SWAP = [IDX[_flip(n)] for n in KP_NAMES]

def mirror(kps):
    m = kps[_SWAP].copy()
    m[:, 0] = -m[:, 0]
    return m

def _group_mean(d):
    ok = ~np.isnan(d)
    cnt = ok.sum(1)
    mean = np.where(ok, d, 0).sum(1) / np.maximum(cnt, 1)
    return np.where(cnt / d.shape[1] >= MIN_SHARED_FRAC, mean, np.nan)

def distances_to_db(q_ang, q_vec, db_ang, db_vec):
    """Pose distance (degrees) from one query pose to every pose in the database. NaN = not comparable."""
    d_ang = np.abs(db_ang - q_ang)
    d_vec = np.degrees(np.arccos(np.clip((db_vec * q_vec).sum(-1), -1, 1)))
    num = np.zeros(len(db_ang)); den = np.zeros(len(db_ang))
    for g, w in ((_group_mean(d_ang), W_ANGLE), (_group_mean(d_vec), W_VEC)):
        ok = ~np.isnan(g)
        num += np.where(ok, g * w, 0); den += np.where(ok, w, 0)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

class PoseDB:
    def __init__(self, cache):
        self.cache = Path(cache)
        z = np.load(self.cache / "poses.npz", allow_pickle=False)
        self.kps, self.images, self.overlays = z["kps"], z["images"], z["overlays"]
        self.backend = str(z["backend"])
        feats = [pose_features(k) for k in self.kps]
        self.ang = np.stack([f[0] for f in feats])
        self.vec = np.stack([f[1] for f in feats])

    def query(self, kps):
        a, v = pose_features(kps)
        d = distances_to_db(a, v, self.ang, self.vec)
        if MIRROR_INVARIANT:
            am, vm = pose_features(mirror(kps))
            d = np.fmin(d, distances_to_db(am, vm, self.ang, self.vec))    # fmin ignores NaN
        return d


# ------------------------------------------------------------------ drawing
PALETTE = [(255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 85),
           (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (85, 0, 255), (170, 0, 255), (255, 0, 170)]

def draw_pose(img, kps):
    out = img.copy()
    t = max(2, int(round(max(out.shape[:2]) / 250)))
    for n, (a, b) in enumerate(list(BONES.values()) + FACE_LINKS):
        pa, pb = kps[IDX[a]], kps[IDX[b]]
        if not (np.isnan(pa).any() or np.isnan(pb).any()):
            cv2.line(out, tuple(map(int, pa)), tuple(map(int, pb)), PALETTE[n % len(PALETTE)], t, cv2.LINE_AA)
    for p in kps:
        if not np.isnan(p).any():
            cv2.circle(out, tuple(map(int, p)), t + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, tuple(map(int, p)), t, (255, 255, 255), -1, cv2.LINE_AA)
    return out


# ------------------------------------------------------------------ step 1: preprocess artworks
def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))             # only ever shrinks, keeps aspect ratio
    return np.array(img)

def preprocess(art_dir, cache, backend):
    art_dir, cache = Path(art_dir), Path(cache)
    (cache / "images").mkdir(parents=True, exist_ok=True)
    (cache / "overlays").mkdir(exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in art_dir.rglob("*") if p.suffix.lower() in exts and not p.name.startswith("."))
    print(f"{len(files)} images found in {art_dir}")

    kps_all, images, overlays = [], [], []
    for n, path in enumerate(files):
        try:
            img = load_image(path)
            people = detect_people(img, backend)
        except Exception as e:
            print(f"  error on {path.name}: {e}")
            continue
        if not people:
            print(f"  no person found: {path.name}")
            continue
        people.sort(key=lambda k: np.nanmean(k[:, 0]))              # label left to right
        base = f"{n:04d}_{path.stem}"
        Image.fromarray(img).save(cache / "images" / f"{base}.jpg", quality=92)
        for k, kps in enumerate(people):
            ov_name = f"{base}__p{k + 1}.jpg"
            Image.fromarray(draw_pose(img, kps)).save(cache / "overlays" / ov_name, quality=92)   # one overlay per skeleton
            kps_all.append(kps); images.append(f"{base}.jpg"); overlays.append(ov_name)
        print(f"  {path.name}: {len(people)} person(s)")

    if not kps_all:
        sys.exit("No poses found: nothing to save.")
    np.savez(cache / "poses.npz", kps=np.stack(kps_all), images=np.array(images),
             overlays=np.array(overlays), backend=np.array(backend))
    print(f"Saved {len(kps_all)} poses to {cache}/")


# ------------------------------------------------------------------ step 2: live web app
@lru_cache(maxsize=128)
def _load(path):
    return np.array(Image.open(path).convert("RGB"))

def choose_match(d, current):
    """Best database index, sticking with `current` unless something is clearly better. None if no match."""
    if np.all(np.isnan(d)):
        return None
    best = int(np.nanargmin(d))
    if current is not None and not np.isnan(d[current]) and d[current] <= d[best] + HYSTERESIS_DEG:
        return current
    return best



def run(cache, backend, port, share):
    from flask import Flask, jsonify, request, send_file, render_template_string

    db = PoseDB(cache)
    app = Flask(__name__)

    # MediaPipe 33 landmarks -> your OpenPose-18 format
    MP_MAP = {
        0: 0,    # nose
        12: 2,   # right shoulder
        14: 3,   # right elbow
        16: 4,   # right wrist
        11: 5,   # left shoulder
        13: 6,   # left elbow
        15: 7,   # left wrist
        24: 8,   # right hip
        26: 9,   # right knee
        28: 10,  # right ankle
        23: 11,  # left hip
        25: 12,  # left knee
        27: 13,  # left ankle
        5: 14,   # right eye
        2: 15,   # left eye
        8: 16,   # right ear
        7: 17    # left ear
    }

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    @app.route("/match", methods=["POST"])
    def match():
        data = request.get_json(silent=True)

        if not data or "keypoints" not in data:
            return jsonify({"error": "Missing keypoints"}), 400

        raw = data["keypoints"]

        if not isinstance(raw, list) or len(raw) != 18:
            return jsonify({"error": "Expected 18 keypoints"}), 400

        try:
            kps = np.array([
                [
                    np.nan if p[0] is None else float(p[0]),
                    np.nan if p[1] is None else float(p[1])
                ]
                for p in raw
            ], dtype=float)
        except (TypeError, ValueError, IndexError):
            return jsonify({"error": "Invalid keypoints"}), 400

        if np.isinf(kps).any():
            return jsonify({"error": "Invalid coordinates"}), 400

        if np.sum(~np.isnan(kps[:, 0])) < MIN_KEYPOINTS:
            return jsonify({"error": "Not enough visible joints"}), 200

        d = db.query(kps)

        if np.all(np.isnan(d)):
            return jsonify({"error": "Pose not comparable"}), 200

        idx = int(np.nanargmin(d))
        distance = float(d[idx])

        return jsonify({
            "idx": idx,
            "distance": round(distance, 2),
            "similarity": round(max(0, 100 - distance), 1),
            "filename": str(db.images[idx])
        })

    @app.route("/artwork/<int:idx>")
    def artwork(idx):
        if idx < 0 or idx >= len(db.images):
            return "Artwork not found", 404

        path = db.cache / "images" / str(db.images[idx])
        return send_file(path, mimetype="image/jpeg")

    @app.route("/overlay/<int:idx>")
    def overlay(idx):
        if idx < 0 or idx >= len(db.overlays):
            return "Overlay not found", 404

        path = db.cache / "overlays" / str(db.overlays[idx])
        return send_file(path, mimetype="image/jpeg")

    PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Living Pose Gallery</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #101010;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

header { padding: 18px; }

h1 {
    font-size: 26px;
    font-weight: 400;
    letter-spacing: 2px;
}

.container {
    width: min(1100px, 95%);
    margin: auto;
}

.stage {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 15px;
    align-items: start;
}

.panel {
    background: #1c1c1c;
    border-radius: 12px;
    overflow: hidden;
    padding: 10px;
    min-width: 0;
}

.camera, .image-container {
    width: 100%;
    aspect-ratio: 3 / 4;
    position: relative;
    background: #111;
    overflow: hidden;
}

video, canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

canvas {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
}

.image-container img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

@media(max-width: 800px) {
    .stage {
        grid-template-columns: 1fr;
    }

    .camera, .image-container {
        aspect-ratio: 4 / 3;
    }
}
.panel h3 {
    font-weight: 400;
    color: #aaa;
}

#artwork {
    width: 100%;
    height: 480px;
    object-fit: contain;
    display: block;
    transition: opacity 0.2s ease;
}

#status {
    color: #50d6bd;
    padding: 20px;
    min-height: 65px;
}

button {
    background: #50d6bd;
    border: 0;
    padding: 13px 30px;
    border-radius: 25px;
    font-size: 16px;
    cursor: pointer;
    margin: 15px;
}

</style>
</head>

<body>

<header>
    <h1>LIVING POSE GALLERY</h1>
    <p>Move your body. Discover your matching artwork.</p>
</header>

<div class="container">

    
    <div class="stage">

        <!-- 1. LIVE VIDEO -->
        <div class="panel">
            <h3>LIVE BODY TRACKING</h3>

            <div class="camera">
                <video id="video" autoplay playsinline muted></video>
                <canvas id="canvas"></canvas>
            </div>
        </div>

        <!-- 2. ARTWORK WITH SKELETON -->
        <div class="panel">
            <h3>POSE MATCH</h3>

            <div class="image-container">
                <img id="overlay" alt="Artwork with skeleton">
            </div>
        </div>

        <!-- 3. ORIGINAL ARTWORK -->
        <div class="panel">
            <h3>ORIGINAL ARTWORK</h3>

            <div class="image-container">
                <img id="artwork" alt="Original artwork">
            </div>
        </div>

    </div>


    <div id="status">Initializing camera...</div>

    <button id="start">START EXPERIENCE</button>

</div>

<script type="module">

import {
    FilesetResolver,
    PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const artwork = document.getElementById("artwork");
const overlay = document.getElementById("overlay");
const status = document.getElementById("status");
const startButton = document.getElementById("start");

let landmarker = null;
let frameCount = 0;
let lastVideoTime = -1;
let currentArtwork = -1;
let requestPending = false;

const MP_MAP = {
    0: 0,
    12: 2,
    14: 3,
    16: 4,
    11: 5,
    13: 6,
    15: 7,
    24: 8,
    26: 9,
    28: 10,
    23: 11,
    25: 12,
    27: 13,
    5: 14,
    2: 15,
    8: 16,
    7: 17
};

const CONNECTIONS = [
    [11,12],
    [11,13], [13,15],
    [12,14], [14,16],
    [11,23], [12,24],
    [23,24],
    [23,25], [25,27],
    [24,26], [26,28],
    [0,11], [0,12]
];

async function initialize() {

    status.textContent = "Loading MediaPipe...";

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );

    landmarker = await PoseLandmarker.createFromOptions(
        vision,
        {
            baseOptions: {
                modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        }
    );

    status.textContent = "MediaPipe ready. Starting camera...";

    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: 640,
            height: 480,
            facingMode: "user"
        },
        audio: false
    });

    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    status.textContent = "Move into view to find your matching artwork.";

    requestAnimationFrame(processFrame);
}

function convertLandmarks(landmarks) {

    const points = Array.from(
        {length: 18},
        () => [null, null]
    );

    for (const [mpIndex, opIndex] of Object.entries(MP_MAP)) {

        const p = landmarks[Number(mpIndex)];

        if (p && (p.visibility ?? 1) >= 0.3) {
            points[opIndex] = [p.x, p.y];
        }
    }

    // Neck = midpoint between shoulders
    const right = points[2];
    const left = points[5];

    if (right[0] !== null && left[0] !== null) {
        points[1] = [
            (right[0] + left[0]) / 2,
            (right[1] + left[1]) / 2
        ];
    }

    return points;
}

function drawSkeleton(landmarks) {

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks) return;

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#50d6bd";
    ctx.fillStyle = "#ffffff";

    for (const [a, b] of CONNECTIONS) {

        const p1 = landmarks[a];
        const p2 = landmarks[b];

        if ((p1.visibility ?? 1) < 0.3 ||
            (p2.visibility ?? 1) < 0.3) continue;

        ctx.beginPath();
        ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
        ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
        ctx.stroke();
    }

    for (const p of landmarks) {

        if ((p.visibility ?? 1) < 0.3) continue;

        ctx.beginPath();
        ctx.arc(
            p.x * canvas.width,
            p.y * canvas.height,
            4, 0, Math.PI * 2
        );
        ctx.fill();
    }
}

async function findArtwork(points) {

    if (requestPending) return;

    requestPending = true;

    try {

        const response = await fetch("/match", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                keypoints: points
            })
        });

        const result = await response.json();

        if (result.idx !== undefined) {

            if (result.idx !== currentArtwork) {

                artwork.style.opacity = 0;
                overlay.style.opacity = 0;

                setTimeout(() => {

                    artwork.src = "/artwork/" + result.idx;
                    overlay.src = "/overlay/" + result.idx;

                    artwork.style.opacity = 1;
                    overlay.style.opacity = 1;

                }, 150);

                currentArtwork = result.idx;
            }

            status.textContent =
                "Closest artwork: " + result.filename +
                " | Distance: " + result.distance + "°" +
                " | Similarity: " + result.similarity + "%";

        } else {
            status.textContent = result.error || "Waiting for pose...";
        }

    } catch (error) {
        status.textContent = "Matching error: " + error.message;
    }

    requestPending = false;
}

function processFrame() {

    if (video.readyState >= 2 &&
        video.currentTime !== lastVideoTime) {

        lastVideoTime = video.currentTime;

        const result = landmarker.detectForVideo(
            video,
            performance.now()
        );

        const landmarks = result.landmarks?.[0];

        drawSkeleton(landmarks);

        frameCount++;

        // Match every fifth video frame
        if (frameCount % 5 === 0 && landmarks) {

            const points = convertLandmarks(landmarks);

            findArtwork(points);
        }

        if (!landmarks) {
            status.textContent = "No body detected. Step into view.";
        }
    }

    requestAnimationFrame(processFrame);
}

startButton.addEventListener("click", async () => {

    startButton.disabled = true;

    try {
        await initialize();
        startButton.textContent = "EXPERIENCE RUNNING";
    } catch (error) {
        status.textContent = "Initialization error: " + error.message;
        startButton.disabled = false;
    }
});

</script>

</body>
</html>
"""

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        threaded=True
    )



# ------------------------------------------------------------------ command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preprocess", help="detect poses in the artworks and build the cache")
    p.add_argument("art_dir"); p.add_argument("--cache", default="./cache")
    p.add_argument("--backend", choices=["yolo", "openpose", "mediapipe"], default="yolo")
    r = sub.add_parser("run", help="start the webcam web app")
    r.add_argument("--cache", default="./cache"); r.add_argument("--port", type=int, default=7860)
    r.add_argument("--backend", choices=["yolo", "openpose", "mediapipe"], default=None,
                   help="live detector (default: same as used for preprocessing)")
    r.add_argument("--share", action="store_true", help="public https link (needed to use the webcam from another device)")
    a = ap.parse_args()
    if a.cmd == "preprocess":
        preprocess(a.art_dir, a.cache, a.backend)
    else:
        run(a.cache, a.backend, a.port, a.share)

if __name__ == "__main__":
    main()
