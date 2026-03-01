import unittest
import os
import cv2
import tqdm
from src.video_reader.video_reader import VideoReader

class TestVideoReader(unittest.TestCase):

    def setUp(self):
        self.video_path = "test_video.mp4" # Replace with a path to a dummy video for testing
        # Create a dummy frame image for testing
        self.frame_path = "test_frame.jpg"
        dummy_frame = (255, 0, 0) # Red frame
        cv2.imwrite(self.frame_path, dummy_frame)

        # Create a dummy video file for testing
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 20.0, (640, 480))
        for _ in range(100):
            out.write(cv2.imread(self.frame_path))
        out.release()
        
    def tearDown(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        if os.path.exists(self.frame_path):
            os.remove(self.frame_path)
        if os.path.exists("output.jpg"):
            os.remove("output.jpg")

    def test_extract_frame(self):
        video_reader = VideoReader(video_path=self.video_path)
        extracted_frame = video_reader.extract_frame(0)
        self.assertIsNotNone(extracted_frame)
        video_reader.release()

    def test_iterate_frames(self):
        video_reader = VideoReader(video_path=self.video_path, iter_start_frame=0, freq=1)
        frames = []
        for frame in video_reader:
            frames.append(frame)
        self.assertEqual(len(frames), 100) # Assuming 100 frames in the dummy video
        self.assertIsNotNone(frames[0])
        video_reader.release()

        video_reader_freq = VideoReader(video_path=self.video_path, iter_start_frame=0, freq=2)
        frames_freq = []
        for frame in video_reader_freq:
            frames_freq.append(frame)
        self.assertEqual(len(frames_freq), 50) # Half the frames with freq=2
        video_reader_freq.release()

        video_reader_start = VideoReader(video_path=self.video_path, iter_start_frame=50, freq=1)
        frames_start = []
        for frame in video_reader_start:
            frames_start.append(frame)
        self.assertEqual(len(frames_start), 50) # Starting from frame 50
        video_reader_start.release()

def main(args):

    video_reader = VideoReader(
        video_path=args.video_path,
        iter_start_frame=args.start_frame,
        freq=args.freq
        )

    if args.is_test:
        print(video_reader)
        total = (video_reader.total_frame - args.start_frame + args.freq - 1) / args.freq
        for frame in tqdm.tqdm(video_reader, initial=args.start_frame, total=total, leave=True):
            pass
    
    import os
    extracted_frame = video_reader.extract_frame(args.extract_frame)
    if not os.path.dirname(args.output_path) == "":
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if os.path.exists(args.output_path):
        raise FileExistsError(f"Error: The output path already exists: {args.output_path}. Please delete/move it or specify a different output path.")
    if extracted_frame is None:
        raise ValueError(f"Error: Failed to extract frame {args.extract_frame} from video.")
    cv2.imwrite(args.output_path, extracted_frame)

    video_reader.release()


if __name__ == "__main__":
    import argparse
    import tqdm

    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video-path", type=str, help="Path to the video file.")
    parser.add_argument("--output-path", type=str, default="output.jpg", help="Path to save extracted frames.")
    parser.add_argument("--extract-frame", type=int, default=0, help="Frame number to start extraction.")

    parser.add_argument("--is-test", action="store_true", help="Run a test.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame number to start iteration.")
    parser.add_argument("--freq", type=int, default=1, help="Frame step size for iteration.")

    args = parser.parse_args()
    main(args)
