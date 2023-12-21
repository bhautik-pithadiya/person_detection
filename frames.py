import subprocess
import os
import cv2


def extract_frames(video_path, start_time, end_time, output_folder,x):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get frames per second (fps) and total number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate start and end frame indices based on time interval
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)

    # Set the video capture object to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read and save frames within the specified interval
    temp = 0
    frame_number = start_frame
    while frame_number <= end_frame:
        temp+=1
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame to the output folder
        if frame_number%30 == 0:
            output_path = f"{output_folder}/frame_{x}_{frame_number:04d}.jpg"
            cv2.imwrite(output_path, frame)

        # Increment the frame number
        frame_number += 1

    # Release the video capture object
    cap.release()


def extract_frames_ffmpeg(video_path, start_time, end_time, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the ffmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-vf', 'fps=1',  # Output one frame per second
        f'{output_folder}/frame_5_%04d.jpg'
    ]

    # Run the ffmpeg command
    subprocess.run(ffmpeg_cmd)
# Example usage:
x = '8'
video_path = f'./Dataset/{x}.mp4'
start_time = 3*60# 40 minutes 41:05
end_time = 3*60 + 34 # 40 minutes and 10 seconds 41:46
output_folder = f"./output_frames/{x}th"

extract_frames(video_path, start_time, end_time, output_folder, x)
# extract_frames_ffmpeg(video_path, start_time, end_time, output_folder)
