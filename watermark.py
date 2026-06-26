import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import time


class VideoWatermarker:
    """Устойчивая к сжатию стеганография для защиты видеозаписей."""

    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    BLOCK_SIZE = 16  
    THRESHOLD = 15   

    def __init__(self, secret_key: str = "INTEGRA_S_2026"):
        self.secret_key = secret_key

    def get_video_files(self, folder_path: str) -> List[Path]:
        video_files = []
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        for file_path in folder.iterdir():
            if file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                video_files.append(file_path)

        return sorted(video_files)

    def check_already_processed(self, input_path: Path, output_folder: Path) -> bool:
        output_filename = f"{input_path.stem}_wm{input_path.suffix}"
        output_path = output_folder / output_filename
        return output_path.exists()

    def embed_watermark(self, input_path: str, output_path: str) -> None:
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        watermark_data = self._prepare_watermark_data(total_frames)
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            watermarked_frame = self._embed_in_frame(frame, watermark_data)
            out.write(watermarked_frame)
            processed_frames += 1

            if processed_frames % 30 == 0:
                print(f"  Processed: {processed_frames}/{total_frames}")

        cap.release()
        out.release()

    def verify_watermark(self, video_path: str) -> Tuple[bool, str]:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return False, "Cannot open video"

        frames_to_check = min(5, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        extracted_bits = []

        for _ in range(frames_to_check):
            ret, frame = cap.read()
            if not ret:
                break
            bits = self._extract_from_frame(frame)
            if not extracted_bits:
                extracted_bits = bits

        cap.release()

        if not extracted_bits:
            return False, "No frames to check"

        try:
            extracted_data = self._decode_data(extracted_bits)
            if self.secret_key in extracted_data:
                return True, extracted_data
            return False, f"Invalid key. Got: {extracted_data}"
        except Exception as e:
            return False, f"Decoding error: {str(e)}"

    def _prepare_watermark_data(self, total_frames: int) -> List[int]:
        timestamp = str(int(time.time()))
        data = f"{self.secret_key}|{timestamp}|{total_frames}"

        bits = []
        for char in data:
            bits.extend([int(b) for b in format(ord(char), '08b')])

        return bits

    def _embed_in_frame(self, frame: np.ndarray, watermark_bits: List[int]) -> np.ndarray:
        b_channel = frame[:, :, 0].astype(np.float32)
        h, w = b_channel.shape
        
        blocks_x = w // self.BLOCK_SIZE
        blocks_y = h // self.BLOCK_SIZE
        
        for i, bit in enumerate(watermark_bits):
            if i >= blocks_x * blocks_y:
                break
                
            block_y = (i // blocks_x) * self.BLOCK_SIZE
            block_x = (i % blocks_x) * self.BLOCK_SIZE
            
            block = b_channel[block_y:block_y+self.BLOCK_SIZE, block_x:block_x+self.BLOCK_SIZE]
            
            half_size = self.BLOCK_SIZE // 2
            left_half = block[:, :half_size]
            right_half = block[:, half_size:]
            
            mean_left = np.mean(left_half)
            mean_right = np.mean(right_half)
            diff = mean_left - mean_right
            
            if bit == 1:
                if diff < self.THRESHOLD:
                    shift = self.THRESHOLD - diff + 2
                    left_half += shift
            else:
                if diff > -self.THRESHOLD:
                    shift = diff + self.THRESHOLD + 2
                    left_half -= shift
                    
            block[:, :half_size] = np.clip(left_half, 0, 255)
            block[:, half_size:] = np.clip(right_half, 0, 255)
            
            b_channel[block_y:block_y+self.BLOCK_SIZE, block_x:block_x+self.BLOCK_SIZE] = block

        frame[:, :, 0] = b_channel.astype(np.uint8)
        return frame

    def _extract_from_frame(self, frame: np.ndarray) -> List[int]:
        b_channel = frame[:, :, 0].astype(np.float32)
        h, w = b_channel.shape
        
        blocks_x = w // self.BLOCK_SIZE
        blocks_y = h // self.BLOCK_SIZE
        
        bits = []
        
        for i in range(min(400, blocks_x * blocks_y)):
            block_y = (i // blocks_x) * self.BLOCK_SIZE
            block_x = (i % blocks_x) * self.BLOCK_SIZE
            
            block = b_channel[block_y:block_y+self.BLOCK_SIZE, block_x:block_x+self.BLOCK_SIZE]
            
            half_size = self.BLOCK_SIZE // 2
            left_half = block[:, :half_size]
            right_half = block[:, half_size:]
            
            mean_left = np.mean(left_half)
            mean_right = np.mean(right_half)
            
            if mean_left > mean_right:
                bits.append(1)
            else:
                bits.append(0)

        return bits

    def _decode_data(self, bits: List[int]) -> str:
        chars = []
        for i in range(0, len(bits) - 7, 8):
            byte = bits[i:i+8]
            char_code = int(''.join(map(str, byte)), 2)
            if 32 <= char_code <= 126:
                chars.append(chr(char_code))

        return ''.join(chars)

    def process_folder(self, input_folder: str, output_folder: str) -> None:
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        output_path.mkdir(parents=True, exist_ok=True)

        video_files = self.get_video_files(input_folder)

        if not video_files:
            print(f"No video files found in {input_folder}")
            return

        print(f"Found {len(video_files)} video(s)")
        print("=" * 60)

        processed_count = 0
        skipped_count = 0

        for video_file in video_files:
            print(f"\nFile: {video_file.name}")

            if self.check_already_processed(video_file, output_path):
                print("  [SKIP] Already processed")
                skipped_count += 1
                continue

            try:
                output_filename = f"{video_file.stem}_wm{video_file.suffix}"
                output_file_path = output_path / output_filename

                print(f"  [PROCESS] Embedding robust watermark...")
                self.embed_watermark(str(video_file), str(output_file_path))

                print(f"  [VERIFY] Checking watermark integrity...")
                valid, data = self.verify_watermark(str(output_file_path))
                if valid:
                    print(f"  [SUCCESS] Watermark verified: {data}")
                    processed_count += 1
                else:
                    print(f"  [FAILED] Verification failed: {data}")

            except Exception as e:
                print(f"  [ERROR] {str(e)}")

        print("\n" + "=" * 60)
        print(f"Completed: {processed_count} processed, {skipped_count} skipped")
        print(f"Output folder: {output_path.absolute()}")
        print("=" * 60)


if __name__ == "__main__":
    watermarker = VideoWatermarker()
    watermarker.process_folder("materials", "materials_with_watermark")