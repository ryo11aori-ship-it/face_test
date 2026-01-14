#!/usr/bin/env python3
"""
CLI entrypoint for face_matcher.

Modes:
- mode1: anchor vs targets (ranking)
- mode2: pair detailed analysis
- mode3: targets-only: find most similar pair
"""
import argparse
import sys
import os
from pathlib import Path
import math
import numpy as np

from face_matcher.core.similarity import cosine_similarity, pairwise_similarities
from face_matcher.utils.image_loader import list_image_files, load_image_pil
from face_matcher.core.detector import detect_best_face
from face_matcher.core.embedding import get_embedding_for_face

def mode1(anchor_path: Path, targets_dir: Path):
    print("[Mode 1] Anchor vs Targets Ranking")
    anchor_img = load_image_pil(anchor_path)
    anchor_face = detect_best_face(anchor_img, return_aligned=False)
    if anchor_face is None:
        print(f"ERROR: no face detected in anchor image: {anchor_path}")
        return 1
    anchor_emb = get_embedding_for_face(anchor_face)

    targets = list_image_files(targets_dir)
    if not targets:
        print("No target images found.")
        return 1

    results = []
    for t in targets:
        img = load_image_pil(t)
        face = detect_best_face(img, return_aligned=False)
        if face is None:
            # skip but report
            results.append((t.name, None, "no_face"))
            continue
        emb = get_embedding_for_face(face)
        sim = float(cosine_similarity(anchor_emb, emb))
        results.append((t.name, sim, "ok"))

    # sort by sim desc, put None at end
    results_sorted = sorted([r for r in results if r[1] is not None],
                            key=lambda x: x[1], reverse=True) + [r for r in results if r[1] is None]

    for rank, item in enumerate(results_sorted, start=1):
        name, sim, status = item
        if status != "ok":
            print(f"{rank:02d}. {name:20s}  -> {status}")
        else:
            print(f"{rank:02d}. {name:20s}  similarity={sim:.3f}")
    return 0

def mode2(img1_path: Path, img2_path: Path):
    print("[Mode 2] Detailed pair analysis")
    img1 = load_image_pil(img1_path)
    img2 = load_image_pil(img2_path)

    face1 = detect_best_face(img1, return_aligned=False, return_landmarks=True)
    face2 = detect_best_face(img2, return_aligned=False, return_landmarks=True)

    if face1 is None:
        print(f"ERROR: no face in {img1_path}")
        return 1
    if face2 is None:
        print(f"ERROR: no face in {img2_path}")
        return 1

    emb1 = get_embedding_for_face(face1.image)
    emb2 = get_embedding_for_face(face2.image)
    sim = float(cosine_similarity(emb1, emb2))

    print(f"Similarity (cosine): {sim:.6f}")

    # Face detection confidences
    print("Face detection confidences:")
    print(f"- {img1_path.name}: {face1.confidence:.3f}")
    print(f"- {img2_path.name}: {face2.confidence:.3f}")

    # Pose approximations (roll via eyes, yaw via eye-nose geometry)
    def approx_pose(landmarks):
        # landmarks: array [[x_left_eye,y_left_eye],[x_right_eye,y_right_eye],[x_nose,y_nose],...]
        le = np.array(landmarks[0])
        re = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        # roll: slope of eye line
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        roll_rad = math.atan2(dy, dx)
        roll = math.degrees(roll_rad)
        # simple yaw proxy: normalized horizontal offset of nose from eye midpoint
        eye_mid = (le + re) / 2.0
        yaw_ratio = (nose[0] - eye_mid[0]) / max(1.0, np.linalg.norm(re - le))
        yaw_deg = float(np.clip(yaw_ratio * 60.0, -90.0, 90.0))  # heuristic scaling
        # pitch proxy (very rough): vertical offset of nose from eye midpoint relative to inter-eye dist
        pitch_ratio = (eye_mid[1] - nose[1]) / max(1.0, np.linalg.norm(re - le))
        pitch_deg = float(np.clip(pitch_ratio * 50.0, -90.0, 90.0))
        return {"yaw": yaw_deg, "pitch": pitch_deg, "roll": roll}

    pose1 = approx_pose(face1.landmarks)
    pose2 = approx_pose(face2.landmarks)
    print("Pose estimates (heuristic):")
    print(f"- {img1_path.name}: yaw={pose1['yaw']:.2f}°, pitch={pose1['pitch']:.2f}°, roll={pose1['roll']:.2f}°")
    print(f"- {img2_path.name}: yaw={pose2['yaw']:.2f}°, pitch={pose2['pitch']:.2f}°, roll={pose2['roll']:.2f}°")

    # landmark distance (normalized)
    def landmarks_distance(lm1, lm2):
        a = np.array(lm1).reshape(-1,2)
        b = np.array(lm2).reshape(-1,2)
        if a.shape != b.shape:
            return float('inf')
        return float(np.mean(np.linalg.norm(a - b, axis=1)))

    lm_dist = landmarks_distance(face1.landmarks, face2.landmarks)
    print(f"Mean landmark L2 distance (pixels): {lm_dist:.3f}")

    # face size ratio
    def face_size(box):
        x1,y1,x2,y2 = box
        return abs(x2 - x1) * abs(y2 - y1)
    fs1 = face_size(face1.bbox)
    fs2 = face_size(face2.bbox)
    print(f"Face size (bbox area): {img1_path.name}={fs1:.1f}, {img2_path.name}={fs2:.1f}")

    # Interpretation hint
    if sim >= 0.80:
        hint = "High probability of same person (similarity >= 0.80)."
    elif sim >= 0.65:
        hint = "Possible match — caution (0.65 <= similarity < 0.80)."
    else:
        hint = "Unlikely same person (similarity < 0.65)."
    print("\nInterpretation (heuristic):")
    print(hint)
    print("\nNote: These outputs are estimates and not definitive identification.")
    return 0

def mode3(targets_dir: Path):
    print("[Mode 3] Find most similar pair among targets")
    targets = list_image_files(targets_dir)
    if len(targets) < 2:
        print("Need at least 2 images in targets_dir.")
        return 1

    faces = []
    names = []
    for t in targets:
        img = load_image_pil(t)
        face = detect_best_face(img, return_aligned=False)
        if face is None:
            continue
        emb = get_embedding_for_face(face.image)
        faces.append(emb)
        names.append(t.name)

    if len(faces) < 2:
        print("Not enough faces detected to compare.")
        return 1

    sims = pairwise_similarities(np.stack(faces, axis=0))
    # upper triangular max
    n = sims.shape[0]
    best = (-1.0, (0,1))
    for i in range(n):
        for j in range(i+1,n):
            s = sims[i,j]
            if s > best[0]:
                best = (s, (i,j))
    s, (i,j) = best
    print("Most similar pair:")
    print(f"- {names[i]}")
    print(f"- {names[j]}")
    print(f"Similarity: {s:.6f}")
    return 0

def build_arg_parser():
    p = argparse.ArgumentParser(description="face_matcher CLI")
    sp = p.add_subparsers(dest="mode", required=True)

    p1 = sp.add_parser("mode1", help="anchor vs targets ranking")
    p1.add_argument("--anchor", required=True, type=Path, help="anchor image path")
    p1.add_argument("--targets_dir", required=True, type=Path, help="directory of target images")

    p2 = sp.add_parser("mode2", help="pair detailed analysis")
    p2.add_argument("--img1", required=True, type=Path)
    p2.add_argument("--img2", required=True, type=Path)

    p3 = sp.add_parser("mode3", help="find most similar pair among targets")
    p3.add_argument("--targets_dir", required=True, type=Path)

    return p

def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "mode1":
            return_code = mode1(args.anchor, args.targets_dir)
        elif args.mode == "mode2":
            return_code = mode2(args.img1, args.img2)
        elif args.mode == "mode3":
            return_code = mode3(args.targets_dir)
        else:
            parser.print_help()
            return_code = 2
    except Exception as e:
        print("Fatal error:", e)
        return_code = 3
    return return_code

if __name__ == "__main__":
    sys.exit(main())
