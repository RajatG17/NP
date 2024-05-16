import os
import torch
import cv2
import argparse
import numpy as np
from models.generator import Generator
from utils.visualize import generate_frames

def generate_video(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator
    generator = Generator()
    generator.load_state_dict(torch.load(os.path.join(args.save_dir, "generator.pth"), map_location=device))
    generator.to(device)
    generator.eval()

    for i in range(args.num_videos):
        # Generate frames
        video_frames, _ = generate_frames(
            generator=generator,
            rgb_depth_pair=(rgb_image, depth_image),
            initial_position=args.initial_position,
            initial_orientation=args.initial_orientation,
            num_frames=args.num_frames,
            delta_degrees=args.delta_degrees,
            device=device,
            save_dir=os.path.join(args.save_dir, f'frames_{i}'),
            attention=args.attention
        )
        # Save frames
        save_frames_dir = os.path.join(args.save_dir, f'frames_{i}')
        os.makedirs(save_frames_dir, exist_ok=True)
        for j, frame in enumerate(video_frames):
            cv2.imwrite(os.path.join(save_frames_dir, f'frame_{j}.png'), (frame * 255).astype('uint8'))

        # Generate video from frames
        height, width, channels = video_frames[0].shape
        video_path = os.path.join(args.save_dir, f'output_{i}.mp4')
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
        for frame in video_frames:
            video.write((frame * 255).astype('uint8'))
        video.release()
        print(f"Generated video: {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Path to directory containing saved models")
    parser.add_argument("--num_videos", type=int, default=1, help="Number of videos to generate")
    parser.add_argument("--initial_position", type=list, default=[0, 0, 0], help="Initial camera position for video generation")
    parser.add_argument("--initial_orientation", type=list, default=[0, 0, 0], help="Initial camera orientation (roll, pitch, yaw) for video generation")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to generate for each video")
    parser.add_argument("--delta_degrees", type=float, default=5.0, help="Degrees to pan the camera view between frames")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the generated videos")

    args = parser.parse_args()
    generate_video(args)
