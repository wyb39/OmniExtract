#!/usr/bin/env python3
"""
Simple file download script
Use urllib to download files; more stable than FTP
"""

import os
import urllib.request
import urllib.error
import time
import logging
from pathlib import Path
import tarfile
import zipfile
import shutil

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, output_subdir="ewas_sup_tables", max_retries=3):
    """
    Download a file
    
    Args:
        url (str): File URL
        output_subdir (str): Subdirectory to create under example_data
        max_retries (int): Maximum retry attempts
    
    Returns:
        str: Downloaded file path, or None if failed
    """
    example_dir = Path(__file__).resolve().parent
    output_dir_path = example_dir / output_subdir
    output_dir_path.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(url)
    output_path = output_dir_path / filename
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Starting download: {url} (attempt {attempt + 1}/{max_retries})")
            urllib.request.urlretrieve(url, str(output_path))
            if output_path.exists():
                file_size = output_path.stat().st_size
                logger.info(f"Download succeeded: {output_path} ({file_size} bytes)")
                try:
                    suffix_concat = "".join(output_path.suffixes)
                    stem = (
                        output_path.name.replace(suffix_concat, "")
                        if suffix_concat
                        else output_path.stem
                    )
                    extract_dir = output_dir_path / stem
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    did_extract = False
                    if tarfile.is_tarfile(str(output_path)):
                        with tarfile.open(str(output_path), "r:*") as tar:
                            tar.extractall(path=str(extract_dir))
                        logger.info(f"Extracted to: {extract_dir}")
                        did_extract = True
                    elif zipfile.is_zipfile(str(output_path)):
                        with zipfile.ZipFile(str(output_path)) as zf:
                            zf.extractall(str(extract_dir))
                        logger.info(f"Extracted to: {extract_dir}")
                        did_extract = True
                    else:
                        logger.warning("Unsupported archive format; skipping extraction")
                    if did_extract:
                        keep_names = {"13229_2018_224_MOESM2_ESM.xlsx", "13229_2018_224_MOESM3_ESM.csv"}
                        moved = 0
                        for root, dirs, files in os.walk(str(output_dir_path)):
                            for f in files:
                                if f in keep_names:
                                    src = Path(root) / f
                                    dst = output_dir_path / f
                                    if src != dst:
                                        try:
                                            shutil.move(str(src), str(dst))
                                            moved += 1
                                        except Exception as me:
                                            logger.error(f"Move failed for {src} -> {dst}: {me}")
                                    else:
                                        moved += 1
                        if moved == 0:
                            logger.warning("Target supplemental files not found after extraction")
                        for entry in output_dir_path.iterdir():
                            if entry.is_file() and entry.name not in keep_names:
                                try:
                                    entry.unlink()
                                except Exception as de:
                                    logger.error(f"Delete failed for file {entry}: {de}")
                            elif entry.is_dir():
                                try:
                                    shutil.rmtree(str(entry))
                                except Exception as de:
                                    logger.error(f"Delete failed for directory {entry}: {de}")
                except Exception as e:
                    logger.error(f"Extraction failed: {e}")
                return str(output_path)
            else:
                logger.warning("File downloaded but not found on disk")
        except urllib.error.URLError as e:
            logger.error(f"URL error (attempt {attempt + 1}): {e}")
        except Exception as e:
            logger.error(f"Download error (attempt {attempt + 1}): {e}")
        if attempt < max_retries - 1:
            wait_time = 5 * (attempt + 1)
            logger.info(f"Retrying after {wait_time} seconds...")
            time.sleep(wait_time)
    
    logger.error(f"All {max_retries} attempts failed")
    return None
    
def main():
    """Main function"""
    file_url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/58/98/PMC6022498.tar.gz"
    logger.info(f"Starting download: {file_url}")
    downloaded_file = download_file(file_url, output_subdir="ewas_sup_tables")
    if downloaded_file:
        logger.info(f"Download successful. Saved to: {downloaded_file}")
        return 0
    else:
        logger.error("Download failed")
        return 1
    
if __name__ == "__main__":
    exit(main())
