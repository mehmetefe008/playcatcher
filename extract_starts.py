import argparse
import glob
import os
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int


def parse_roi(s: str) -> ROI:
    # format: x,y,w,h
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI formatı x,y,w,h olmalı (ör: 0,1200,1170,800)")
    return ROI(*parts)


def parse_scales(s: str) -> list[float]:
    # format: "0.6,0.7,0.8,0.9,1.0,1.1"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    scales = [float(p) for p in parts]
    if not scales:
        raise ValueError("--scales boş olamaz.")
    return scales


def best_match_multiscale(gray: np.ndarray, template_gray: np.ndarray, scales: list[float]):
    """
    gray: tek kanal (uint8) frame
    template_gray: tek kanal template
    scales: template ölçekleri (örn [0.6..1.4])

    return: (best_val, best_loc, best_size, best_scale)
    """
    best_val = -1.0
    best_loc = None
    best_size = None
    best_scale = None

    gh, gw = gray.shape[:2]
    th0, tw0 = template_gray.shape[:2]

    for s in scales:
        # Template'i ölçekle
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
        tpl = cv2.resize(template_gray, (0, 0), fx=s, fy=s, interpolation=interp)

        th, tw = tpl.shape[:2]

        # Template, arama görüntüsünden büyük/eşitse matchTemplate çalışmaz
        if th >= gh or tw >= gw:
            continue

        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            best_val = float(max_val)
            best_loc = max_loc
            best_size = (tw, th)
            best_scale = float(s)

    return best_val, best_loc, best_size, best_scale


def find_hits(
    video_path,
    template_gray,
    sample_fps=20.0,
    threshold=0.5,
    cooldown=5.0,
    roi: ROI | None = None,
    scales: list[float] | None = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else None

    step = max(1, int(round(fps / sample_fps)))

    hits = []
    last_hit_t = -1e9
    frame_idx = 0

    # debug: en iyi eşleşme
    best = (-1.0, 0.0, None)  # (max_val, time, scale)

    # default scales
    if scales is None:
        scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    pbar = tqdm(
        total=frame_count if frame_count > 0 else None,
        desc=os.path.basename(video_path),
        unit="frame",
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            t = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if roi is not None:
                x, y, w, h = roi.x, roi.y, roi.w, roi.h
                gray = gray[y : y + h, x : x + w]

            # Multi-scale template matching
            max_val, max_loc, max_size, max_scale = best_match_multiscale(gray, template_gray, scales)

            # debug: en iyi eşleşmeyi takip et
            if max_val > best[0]:
                best = (max_val, t, max_scale)

            if max_val >= threshold and (t - last_hit_t) >= cooldown:
                hits.append(t)
                last_hit_t = t

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # debug: en iyi skor + ölçek
    if best[2] is None:
        print(f"En iyi eslesme skoru: {best[0]:.3f} @ {best[1]:.3f}s (olcek: N/A)")
    else:
        print(f"En iyi eslesme skoru: {best[0]:.3f} @ {best[1]:.3f}s (olcek: {best[2]:.2f}x)")

    return hits, duration


def extract_frame_ffmpeg(video_path, t_sec, out_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{t_sec:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        out_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help='Örn: "videos/*.mp4" veya tek dosya yolu')
    ap.add_argument("--template", required=True, help="start_template.png yolu")
    ap.add_argument("--out", default="out", help="Çıktı klasörü")
    ap.add_argument("--offset", type=float, default=2.5, help="Start tespitinden kaç saniye sonrası")
    ap.add_argument("--sample-fps", type=float, default=5.0, help="Taramada saniyede kaç kare bakılsın")
    ap.add_argument("--threshold", type=float, default=0.85, help="Eşleşme eşiği (0-1)")
    ap.add_argument("--cooldown", type=float, default=2.0, help="Arka arkaya tespitleri tek saymak için saniye")
    ap.add_argument("--roi", type=str, default=None, help="İsteğe bağlı: x,y,w,h")
    ap.add_argument(
        "--scales",
        type=str,
        default=None,
        help='İsteğe bağlı: "0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4"',
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    template = cv2.imread(args.template, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise RuntimeError("Template okunamadı. --template yolunu kontrol et.")

    roi = parse_roi(args.roi) if args.roi else None
    scales = parse_scales(args.scales) if args.scales else None

    video_list = glob.glob(args.videos)
    if not video_list and os.path.isfile(args.videos):
        video_list = [args.videos]
    if not video_list:
        raise RuntimeError("Video bulunamadı. --videos desenini/yolunu kontrol et.")

    for vp in video_list:
        hits, duration = find_hits(
            vp,
            template,
            sample_fps=args.sample_fps,
            threshold=args.threshold,
            cooldown=args.cooldown,
            roi=roi,
            scales=scales,
        )

        base = os.path.splitext(os.path.basename(vp))[0]

        times_txt = os.path.join(args.out, f"{base}_times.txt")
        with open(times_txt, "w", encoding="utf-8") as f:
            for t in hits:
                f.write(f"{t:.3f}\n")

        for i, t in enumerate(hits, start=1):
            target = t + args.offset
            if duration is not None and target > duration:
                continue
            out_png = os.path.join(args.out, f"{base}_start_{i:03d}_{target:.3f}s.png")
            extract_frame_ffmpeg(vp, target, out_png)

        print(f"\n{base}: {len(hits)} adet start tespiti → ekran görüntüleri çıkarıldı. Zamanlar: {times_txt}\n")


if __name__ == "__main__":
    main()

