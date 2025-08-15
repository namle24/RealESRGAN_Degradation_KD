import os
import cv2
import glob

# Định nghĩa thư mục chứa frame và file video đầu ra
frame_dir = r"D:\DL_for_enhance_video_image_quality\data\VRT_result"
output_video_path = r"D:\DL_for_enhance_video_image_quality\results\output_vrt.mp4"

# Kiểm tra thư mục frame có tồn tại không
if not os.path.isdir(frame_dir):
    print(f"Error: Directory {frame_dir} does not exist!")
else:
    # Lấy danh sách frame
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame*.png")))
    if not frame_files:
        print(f"No frames found in {frame_dir}!")
    else:
        # Đọc frame đầu tiên để lấy kích thước
        first_frame = cv2.imread(frame_files[0])
        height, width = first_frame.shape[:2]

        # Khởi tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
        fps = 30  # FPS của video (thay đổi nếu cần)
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Ghi từng frame vào video
        print("Combining frames into video...")
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            video_writer.write(frame)
            print(f"Added {frame_file}")

        # Giải phóng VideoWriter
        video_writer.release()
        print(f"Video saved successfully at: {output_video_path}")