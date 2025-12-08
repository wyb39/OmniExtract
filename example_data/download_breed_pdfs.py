import csv
import os
import requests
from pathlib import Path
import time
from urllib.parse import urlparse

def download_pdf(url, output_path, max_retries=3):
    """Download PDF file with retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the PDF file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully downloaded: {os.path.basename(output_path)}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                print(f"Failed to download after {max_retries} attempts: {url}")
                return False

def process_csv(csv_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    downloaded_count = 0
    failed_count = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)
            for row in csv_reader:
                if len(row) < 3:
                    print(f"Skipping malformed row: {row}")
                    continue
                breed_name = row[0].strip('"')
                download_link = row[1].strip('"')
                breed_id = row[2].strip('"')
                output_filename = f"{breed_id}_akc_standard.pdf"
                output_path = os.path.join(output_directory, output_filename)
                if os.path.exists(output_path):
                    print(f"Skipping {output_filename} - already exists")
                    downloaded_count += 1
                    continue
                print(f"Downloading: {breed_name} ({breed_id})")
                if download_pdf(download_link, output_path):
                    downloaded_count += 1
                else:
                    failed_count += 1
                time.sleep(1)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        return 0, 0
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        return 0, 0
    print("\nDownload Summary:")
    print(f"Total files processed: {downloaded_count + failed_count}")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Failed downloads: {failed_count}")
    print(f"Files saved to: {os.path.abspath(output_directory)}")
    return downloaded_count, failed_count

def main():
    base_dir = Path(__file__).parent
    orig_csv = base_dir / "breed_pdf_links.csv"
    orig_out = base_dir / "downloaded_breed_pdfs"
    pred_csv = base_dir / "breed_pdf_links_pred.csv"
    pred_out = base_dir / "downloaded_breed_pdfs_pred"

    print("Processing official AKC links...")
    process_csv(str(orig_csv), str(orig_out))

    print("\nProcessing predicted AKC links...")
    process_csv(str(pred_csv), str(pred_out))

if __name__ == "__main__":
    main()
