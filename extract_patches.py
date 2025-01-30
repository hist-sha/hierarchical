import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
import logging
from PIL import Image, ImageFilter, ImageStat
from multiprocessing import Process, JoinableQueue, Manager
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator


Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class TileWorker(Process):
    def __init__(self, queue, slide_path, tile_size, threshold, format, saved_tiles, lock):
        super().__init__()
        self.queue = queue
        self.slide_path = slide_path
        self.tile_size = tile_size
        self.threshold = threshold
        self.format = format
        self.saved_tiles = saved_tiles
        self.lock = lock
        self.slide = None

    def run(self):
        self.slide = open_slide(self.slide_path)
        dz = DeepZoomGenerator(self.slide, self.tile_size, overlap=0, limit_bounds=True)
        
        while True:
            data = self.queue.get()
            if data is None:
                self.queue.task_done()
                break
            level, col, row, tile_path = data
            
            try:
                tile = dz.get_tile(level, (col, row))
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge_value = np.mean(ImageStat.Stat(edge).sum) / (self.tile_size ** 2)
                
                if edge_value > self.threshold:
                    tile.save(tile_path, format=self.format)
                    with self.lock:
                        self.saved_tiles.value += 1
            except Exception as e:
                logger.error(f"Error processing tile {col}_{row}: {e}")
            
            self.queue.task_done()


def extract_patches(slide_path, output_dir="data", levels=(0,), tile_size=224, format='jpeg', threshold=15, workers=10):
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Extraction started")
    if os.path.isdir(slide_path):
        slide_files = glob.glob(os.path.join(slide_path, "*.svs"))
    else:
        slide_files = [slide_path]
    
    for slide_file in sorted(slide_files):
        slide_name = os.path.splitext(os.path.basename(slide_file))[0]
        slide_output = os.path.join(output_dir, slide_name)
        os.makedirs(slide_output, exist_ok=True)
        
        slide = open_slide(slide_file)
        dz = DeepZoomGenerator(slide, tile_size, overlap=0, limit_bounds=True)
        max_level = dz.level_count - 1
        queue = JoinableQueue()
        
        manager = Manager()
        saved_tiles = manager.Value('i', 0)
        lock = manager.Lock()
        
        workers_list = [TileWorker(queue, slide_file, tile_size, threshold, format, saved_tiles, lock) for _ in range(workers)]
        for worker in workers_list:
            worker.start()
        
        for level in levels:
            actual_level = max_level - level
            if actual_level < 0:
                logger.warning(f"Level {level} is out of range for {slide_name}.")
                continue
            
            level_dir = os.path.join(slide_output, f'level{level}')
            os.makedirs(level_dir, exist_ok=True)
            cols, rows = dz.level_tiles[actual_level]
            total_tiles = cols * rows
            logger.info(f"processing {slide_name} lvl {level}")
            
            for row in range(rows):
                for col in range(cols):
                    tile_path = os.path.join(level_dir, f'{col}_{row}.{format}')
                    queue.put((actual_level, col, row, tile_path))

            queue.join()
            with lock:
                logger.info(f"{saved_tiles.value}/{total_tiles} tiles saved after filtering")
                saved_tiles.value = 0

        for _ in workers_list:
            queue.put(None)
        for worker in workers_list:
            worker.join()
    
    logger.info("Extraction completed")


if __name__ == '__main__':
    # python extract_patches.py -i /home/sha/Documents/data/WSS2_v2_stomach/train -o data
    parser = argparse.ArgumentParser(description='Extract patches from WSI')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input WSI file or directory')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('-l', '--levels', type=int, default=[0, 1, 2], help='Levels to extract')
    parser.add_argument('-t', '--tile_size', type=int, default=224, help='Tile size')
    parser.add_argument('-f', '--format', type=str, default='jpeg', help='Image format (jpeg, png)')
    parser.add_argument('-th', '--threshold', type=float, default=5, help='Threshold for filtering background')
    parser.add_argument('-w', '--workers', type=int, default=20, help='Number of worker processes')
    
    args = parser.parse_args()
    extract_patches(args.input, args.output, args.levels, args.tile_size, args.format, args.threshold, args.workers)
