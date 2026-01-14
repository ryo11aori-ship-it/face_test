#!/usr/bin/env python3
"""
CLI entrypoint for face_matcher.

Modes:
- mode1: anchor vs targets (ranking)
- mode2: pair detailed analysis
- mode3: targets-only: find most similar pair
"""
from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path
from typing import Optional

import numpy as np

from face_matcher.core.similarity import cosine_similarity, pairwise_similarities
from face_matcher.utils.image_loader import list_image_files, load_image_pil
from face_matcher.core.detector import detect_best_face
from face_matcher.core.embedding import get_embedding_for_face


def mode1(anchor_path: Path, targets_dir: Path) -> int:
    """
    Mode 1: Given an anchor image and a directory of target images,
    compute similarity score for each target and print ranking.
    """
    print("[Mode 1] Anchor vs Targets Ranking")
    anchor_img = load_image_pil(anchor_path)
    anchor_face = detect_best_face(anchor_img, return_aligned=False)
    if anchor_face is None:
        print(f"ERROR: no face detected in anchor image: {anchor_path}")
        return 1

    # Use the PIL image stored in DetectedFace
    anchor_emb = get_embedding_for_face(anchor_face.image)

    targets = list_image_files(targets_dir)
    if not targets:
        print("No target images found.")
        return 1

    results = []
    for t in targets:
        try:
            img = load_image_pil(t)
        except Exception as e:
            results.append((t.name, None, f"load_error:{e}"))
            continue

        face = detect_best_face(img, return_aligned=False)
        if face is None:
            # skip but report
            results.append((t.name, None, "no_face"))
            continue

        try:
            emb = get_embedding_for_face(face.image)
            sim = float(cosine_similarity(anchor_emb, emb))
            results.append((t.name, sim, "ok"))
        except Exception as e:
            results.append((t.name, None, f"embed_error:{e}"))

    # sort by sim desc, put entries with None score at end preserving their order
    scored = [r for r in results if r[1] is not None]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    nonscored = [r for r in results if r[1] is None]
    results_sorted = scored_sorted + nonscored

    for rank, item in enumerate(results_sorted, start=1):
        name, sim, status = item
        if status != "ok":
            print(f"{rank:02d}. {name:20s}  -> {status}")
        else:
            print(f"{rank:02d}. {name:20s}  similarity={sim:.3f}")
    return 0


def mode2(img1_path: Path, img2_path: Path) -> int:
    """
    Mode 2: Detailed analysis between two images.
    Outputs similarity plus detection confidences, pose heuristics, landmark distances, etc.
    """
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
    print(f"- {img1_path.name}: {getattr(face1, 'confidence', 0.0):.3f}")
    print(f"- {img2_path.name}: {getattr(face2, 'confidence', 0.0):.3f}")

    # Pose approximations (roll via eyes, yaw via eye-nose geometry)
    def approx_pose(landmarks):
        # landmarks: list of [x,y] typically [left_eye, right_eye, nose, left_mouth, right_mouth]
        try:
            le = np.array(landmarks[0])
            re = np.array(landmarks[1])
            nose = np.array(landmarks[2])
        except Exception:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        # roll: slope of eye line
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        roll_rad = math.atan2(dy, dx)
        roll = math.degrees(roll_rad)
        # simple yaw proxy: normalized horizontal offset of nose from eye midpoint
        eye_mid = (le + re) / 2.0
        inter_eye = max(1.0, np.linalg.norm(re - le))
        yaw_ratio = (nose[0] - eye_mid[0]) / inter_eye
        yaw_deg = float(np.clip(yaw_ratio * 60.0, -90.0, 90.0))  # heuristic scaling
        # pitch proxy (very rough): vertical offset of nose from eye midpoint relative to inter-eye dist
        pitch_ratio = (eye_mid[1] - nose[1]) / inter_eye
        pitch_deg = float(np.clip(pitch_ratio * 50.0, -90.0, 90.0))
        return {"yaw": yaw_deg, "pitch": pitch_deg, "roll": roll}

    pose1 = approx_pose(face1.landmarks)
    pose2 = approx_pose(face2.landmarks)
    print("Pose estimates (heuristic):")
    print(f"- {img1_path.name}: yaw={pose1['yaw']:.2f}°, pitch={pose1['pitch']:.2f}°, roll={pose1['roll']:.2f}°")
    print(f"- {img2_path.name}: yaw={pose2['yaw']:.2f}°, pitch={pose2['pitch']:.2f}°, roll={pose2['roll']:.2f}°")

    # landmark distance (normalized)
    def landmarks_distance(lm1, lm2):
        try:
            a = np.array(lm1).reshape(-1, 2)
            b = np.array(lm2).reshape(-1, 2)
        except Exception:
            return float('inf')
        if a.shape != b.shape:
            return float('inf')
        return float(np.mean(np.linalg.norm(a - b, axis=1)))

    lm_dist = landmarks_distance(face1.landmarks, face2.landmarks)
    print(f"Mean landmark L2 distance (pixels): {lm_dist:.3f}")

    # face size ratio
    def face_size(box):
        x1, y1, x2, y2 = box
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


def mode3(targets_dir: Path) -> int:
    """
    Mode 3: Given only a directory of target images, find the most similar pair.
    """
    print("[Mode 3] Find most similar pair among targets")
    targets = list_image_files(targets_dir)
    if len(targets) < 2:
        print("Need at least 2 images in targets_dir.")
        return 1

    faces = []
    names = []
    for t in targets:
        try:
            img = load_image_pil(t)
        except Exception as e:
            # skip unreadable files
            continue
        face = detect_best_face(img, return_aligned=False)
        if face is None:
            continue
        try:
            emb = get_embedding_for_face(face.image)
        except Exception:
            continue
        faces.append(emb)
        names.append(t.name)

    if len(faces) < 2:
        print("Not enough faces detected to compare.")
        return 1

    sims = pairwise_similarities(np.stack(faces, axis=0))
    # find upper-triangular max
    n = sims.shape[0]
    best_score = -1.0
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if s > best_score:
                best_score = s
                best_pair = (i, j)

    i, j = best_pair
    print("Most similar pair:")
    print(f"- {names[i]}")
    print(f"- {names[j]}")
    print(f"Similarity: {best_score:.6f}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
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


def main(argv: Optional[list[str]] = None) -> int:
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