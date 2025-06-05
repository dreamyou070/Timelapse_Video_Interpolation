import os

def make_folder_with_permission(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)
    os.chmod(path, 0o777)  # 폴더 권한을 777로 변경

def main():
    trg_folder = '/workspace/data/diffusion/Framer/TrainData'
    for i in range(14):
        folder_name = f'frame_{str(i).zfill(2)}'
        folder_path = os.path.join(trg_folder, folder_name)
        samples = os.listdir(folder_path)
        for sample in samples:
            org_path = os.path.join(folder_path, sample)
            new_path = os.path.join(folder_path, f'sample_1.png')
            os.rename(org_path, new_path)
if __name__ == "__main__":
    main()
