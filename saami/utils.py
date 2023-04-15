"Utility file for saami package"

def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    progress = min(100, (downloaded / total_size) * 100)
    print(f"\rDownload progress: {progress:.2f}%", end='')