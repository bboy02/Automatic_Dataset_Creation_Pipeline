import os
from icrawler.builtin import BingImageCrawler

# Define output directory
output_dir = "white_bicycles"
os.makedirs(output_dir, exist_ok=True)

# Initialize the image crawler (you can switch to GoogleImageCrawler if needed)
crawler = BingImageCrawler(storage={'root_dir': output_dir})

# Start crawling
crawler.crawl(
    keyword='white bicycle',
    max_num=20,
    min_size=(200, 200),  # Minimum image size (optional)
    file_idx_offset=0     # Start file index from 0
)

print(f"✅ Download complete. Images saved in '{output_dir}/'")
