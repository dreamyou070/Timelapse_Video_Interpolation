import os
import cv2

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
    trg_folder = '/workspace/data/diffusion/customdata/ClimateBenchmark/youtube_data_raw'
    save_folder = '/workspace/data/diffusion/customdata/ClimateBenchmark/youtube_data_raw_split'
    make_folder_with_permission(save_folder)

    disasters = os.listdir(trg_folder)
    for disaster in disasters:
        disaster_path = os.path.join(trg_folder, disaster)
        subjects = os.listdir(disaster_path)
        for subject in subjects:
            subjects_path = os.path.join(disaster_path, subject)
            videos = os.listdir(subjects_path)
            for video in videos:
                video_path = os.path.join(subjects_path, video)
                name, ext = os.path.splitext(video)
                save_video_folder = os.path.join(save_folder, name)
                make_folder_with_permission(save_video_folder)

                # split video into frames and save on save_video_folder
                split_video_into_frames(video_path, save_video_folder)

if __name__ == "__main__":
    main()
