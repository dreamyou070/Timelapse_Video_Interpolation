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

    start_folder = '/workspace/data/diffusion/customdata/ClimateBenchmark/youtube_data_raw_split'
    videos = os.listdir(start_folder)

    save_base_folder = '/workspace/data/diffusion/Framer/TrainData'
    shutil.rmtree(save_base_folder, ignore_errors=True)
    #for video_number, video in enumerate(videos) :
    #    video_path = os.path.join(start_folder, video)
        #frames = os.listdir(video_path)

     #   gap = 10
        #max_frame_num = int(len(frames) / gap)
        #max_frame_num = min(max_frame_num, 50)
        #print(f'max_frame_num = {max_frame_num}')
        #max_frame_num = 140000
        #for i in range(max_frame_num):
        #    src_frame_path = os.path.join(video_path, f'frame_{str(gap*i).zfill(4)}.jpg')
        #    folder_name = f'frame_{str(i).zfill(2)}'
         #   folder_path = os.path.join(save_base_folder, folder_name)
        #    if i

            #if not os.path.exists(folder_path):
                #print(f'make folder {folder_path} ...')
            #    make_folder_with_permission(folder_path)
            #save_path = os.path.join(folder_path, f'sample_{str(video_number).zfill(3)}.png')
            # copy
            #shutil.copyfile(src_frame_path, save_path)


if __name__ == "__main__":
    main()
