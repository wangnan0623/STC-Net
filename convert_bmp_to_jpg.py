import os
from PIL import Image
import shutil

def bmp_2_jpg(root_dir):
    seq_names = sorted(os.listdir(root_dir))

    for seq_name in seq_names:
        bmp_dir = os.path.join(root_dir, seq_name, 'event_imgs')
        jpg_dir = os.path.join(root_dir, seq_name, 'event_imgs_jpg')

        # ✅ 如果已经处理过，跳过
        if not os.path.exists(bmp_dir) and os.path.exists(jpg_dir):
            print(f"Skipping already processed sequence: {seq_name}")
            continue

        # ✅ 如果 vis_imgs 不存在，直接跳过
        if not os.path.exists(bmp_dir):
            print(f"Skipping missing vis_imgs: {seq_name}")
            continue

        # 创建目标目录
        os.makedirs(jpg_dir, exist_ok=True)

        # 转换所有 bmp 为 jpg
        for filename in os.listdir(bmp_dir):
            if filename.endswith(".bmp"):
                source_path = os.path.join(bmp_dir, filename)
                frame_number = filename.replace("frame", "").replace(".bmp", "")
                target_path = os.path.join(jpg_dir, f"{frame_number}.jpg")

                with Image.open(source_path) as img:
                    img.save(target_path, "JPEG")

        print(f"Conversion complete for {seq_name}")

        # 删除原始 bmp 目录
        try:
            shutil.rmtree(bmp_dir)
            print(f"Deleted source directory: {bmp_dir}")
        except Exception as e:
            print(f"Failed to delete {bmp_dir}: {e}")

if __name__ == "__main__":
    root_dir = '/home/Data/VisEvent/train_subset'
    bmp_2_jpg(root_dir)
