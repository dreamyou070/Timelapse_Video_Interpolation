import os
import cv2
import numpy as np
from tqdm import tqdm
import numpy.linalg as LA


def make_folder_with_permission(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)
    os.chmod(path, 0o777)


def compute_optical_flow_farneback(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow  # (H, W, 2)


def cosine_similarity(flow_a, flow_b):
    vec_a = flow_a.reshape(-1, 2)
    vec_b = flow_b.reshape(-1, 2)

    dot_product = np.sum(vec_a * vec_b, axis=1)
    norm_a = LA.norm(vec_a, axis=1) + 1e-6
    norm_b = LA.norm(vec_b, axis=1) + 1e-6

    cos_sim = dot_product / (norm_a * norm_b)
    return np.mean(cos_sim)


def main():

    src_folder = '/workspace/data/diffusion/customdata/ClimateBenchmark/youtube_data_raw_split'
    dst_folder = '/workspace/data/diffusion/Framer/youtube_optical_flows'
    videos = os.listdir(src_folder)

    for video in tqdm(videos):
        video_path = os.path.join(src_folder, video)
        frame_images = sorted([
            f for f in os.listdir(video_path)
            if f.endswith('.png') or f.endswith('.jpg')
        ])

        if len(frame_images) < 3:
            continue  # 최소 세 프레임 필요

        save_video_path = os.path.join(dst_folder, video)
        make_folder_with_permission(save_video_path)

        all_flows = []
        for i in range(len(frame_images) - 1):
            frame_path_1 = os.path.join(video_path, frame_images[i])
            frame_path_2 = os.path.join(video_path, frame_images[i + 1])

            img1 = cv2.imread(frame_path_1)
            img2 = cv2.imread(frame_path_2)

            if img1 is None or img2 is None:
                print(f"⚠️ 프레임 로드 실패: {frame_path_1}, {frame_path_2}")
                continue

            flow = compute_optical_flow_farneback(img1, img2)
            all_flows.append(flow)
            flow_save_path = os.path.join(save_video_path, f'flow_{i:03d}.npy')
            np.save(flow_save_path, flow)

        for i in range(len(all_flows) - 1):
            flow_0 = all_flows[i]
            flow_1 = all_flows[i + 1]
            sim = cosine_similarity(flow_0, flow_1)

            print(f"[{video}] flow_{i:03d} vs flow_{i+1:03d} 유사도: {sim:.4f}")

            if sim < 0.5:  # 이 값은 실험적으로 조정 가능
                print(f"⚠️ 이상 움직임 감지: {video}/flow_{i:03d} ↔ flow_{i+1:03d}")

        # break  # 여러 video 폴더 돌릴 경우 주석 해제

if __name__ == "__main__":
    main()
