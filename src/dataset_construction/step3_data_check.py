import os
import cv2
import shutil

def make_folder_with_permission(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)
    os.chmod(path, 0o777)  # 폴더 권한을 777로 변경

def split_video_into_frames(video_path, save_folder):
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 끝까지 읽었으면 종료
        # 프레임 저장
        frame_filename = f"frame_{str(frame_count).zfill(4)}.jpg"
        frame_path = os.path.join(save_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}.")

def main():

    start_folder = '/workspace/data/diffusion/Framer/TrainData'
    dest_folder = '/workspace/data/diffusion/Framer/TrainDataSequence'
    frames = os.listdir(start_folder)
    for frame in frames:
        frame_path = os.path.join(start_folder, frame)
        videos = os.listdir(frame_path)
        for video in videos:
            name, ext = os.path.splitext(video)
            org_video_path = os.path.join(frame_path, video)
            video_save_folder = os.path.join(dest_folder, name)
            make_folder_with_permission(video_save_folder)
            new_video_path = os.path.join(video_save_folder, f'{frame}.png')
            shutil.copy(org_video_path, new_video_path)

if __name__ == "__main__":
    main()
