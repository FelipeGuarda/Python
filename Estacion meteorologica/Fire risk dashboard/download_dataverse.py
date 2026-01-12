"""
Download datasets from Datos para Resiliencia using pyDataverse.

Installation:
    pip install pyDataverse==0.3.3

Usage examples:
    # Download all files as a single zip
    python download_dataverse.py --download-all

    # Download all files individually
    python download_dataverse.py --download-all --individual

    # Download specific files
    python download_dataverse.py --files "filename1.csv" "filename2.geojson"

Environment variables (optional):
    DATAVERSE_API_KEY: Your API key (otherwise use --api-key flag)
    DATAVERSE_BASE_URL: Base URL (default: https://datospararesiliencia.cl)
    DATAVERSE_DATASET_ID: Dataset DOI (default: doi:10.71578/XAZAKP)
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from pyDataverse.api import NativeApi, DataAccessApi
except ImportError:
    print("ERROR: pyDataverse not found. Install with:")
    print("  pip install pyDataverse==0.3.3")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download datasets from Datos para Resiliencia using pyDataverse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Credentials
    p.add_argument(
        "--api-key",
        default=None,
        help="API key (or set DATAVERSE_API_KEY env var)",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Base URL (default: https://datospararesiliencia.cl or DATAVERSE_BASE_URL env var)",
    )
    
    # Dataset
    p.add_argument(
        "--dataset-id",
        default=None,
        help="Dataset DOI (default: doi:10.71578/XAZAKP or DATAVERSE_DATASET_ID env var)",
    )
    
    # Download options
    p.add_argument(
        "--download-all",
        action="store_true",
        help="Download all files from the dataset",
    )
    p.add_argument(
        "--individual",
        action="store_true",
        help="Download files individually (default: as single zip if --download-all)",
    )
    p.add_argument(
        "--files",
        nargs="+",
        help="Download specific files by name (e.g., --files file1.csv file2.geojson)",
    )
    p.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for downloaded files (default: current directory)",
    )
    p.add_argument(
        "--list-files",
        action="store_true",
        help="List all available files in the dataset (no download)",
    )
    
    return p.parse_args()


def main() -> int:
    args = parse_args()
    
    # Get credentials
    api_key = args.api_key or os.getenv("DATAVERSE_API_KEY")
    base_url = args.base_url or os.getenv("DATAVERSE_BASE_URL", "https://datospararesiliencia.cl")
    dataset_id = args.dataset_id or os.getenv("DATAVERSE_DATASET_ID", "doi:10.71578/XAZAKP")
    
    if not api_key:
        print("ERROR: Missing API key.")
        print("Provide either:")
        print(" - flag: --api-key YOUR_API_KEY")
        print(" - or env var: DATAVERSE_API_KEY")
        return 2
    
    # Mask API key in output
    masked_key = api_key[:8] + "..." if len(api_key) > 8 else "***"
    print(f"Base URL: {base_url}")
    print(f"API Key: {masked_key}")
    print(f"Dataset ID: {dataset_id}")
    print()
    
    # Initialize API clients
    try:
        api = NativeApi(base_url, api_key)
        data_api = DataAccessApi(base_url, api_key)
        print("✓ API clients initialized")
    except Exception as e:
        print(f"ERROR initializing API clients: {e}")
        return 1
    
    # Get dataset metadata
    try:
        print(f"Fetching dataset metadata...")
        dataset = api.get_dataset(dataset_id)
        dataset_json = dataset.json()
        
        if dataset.status_code != 200:
            print(f"ERROR: Failed to fetch dataset (status {dataset.status_code})")
            print(f"Response: {dataset_json}")
            return 1
        
        files_list = dataset_json['data']['latestVersion']['files']
        print(f"✓ Found {len(files_list)} file(s) in dataset")
        print()
    except Exception as e:
        print(f"ERROR fetching dataset: {e}")
        return 1
    
    # List files if requested
    if args.list_files or (not args.download_all and not args.files):
        print("Available files:")
        for i, file in enumerate(files_list, 1):
            filename = file["dataFile"]["filename"]
            file_id = file["dataFile"]["id"]
            file_size = file["dataFile"].get("filesize", "unknown")
            print(f"  {i}. {filename} (ID: {file_id}, size: {file_size} bytes)")
        
        if not args.download_all and not args.files:
            print()
            print("Use --download-all to download all files, or --files to download specific files.")
        return 0
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download all files
    if args.download_all:
        if args.individual:
            # Download each file individually
            print(f"Downloading {len(files_list)} file(s) individually...")
            for file in files_list:
                filename = file["dataFile"]["filename"]
                file_id = file["dataFile"]["id"]
                print(f"  Downloading: {filename} (ID: {file_id})")
                
                try:
                    response = data_api.get_datafile(file_id, is_pid=False)
                    output_path = output_dir / filename
                    
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    print(f"    ✓ Saved to: {output_path}")
                except Exception as e:
                    print(f"    ERROR: {e}")
        else:
            # Download all files as a single zip
            print("Downloading all files as single zip...")
            ids = [file["dataFile"]["id"] for file in files_list]
            
            try:
                response = data_api.get_datafiles(','.join(map(str, ids)))
                output_path = output_dir / "dataset_files.zip"
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                print(f"✓ Saved to: {output_path}")
            except Exception as e:
                print(f"ERROR: {e}")
                return 1
    
    # Download specific files
    elif args.files:
        d_files = args.files
        print(f"Downloading {len(d_files)} specific file(s)...")
        
        # Filter files
        files_to_download = [
            file for file in files_list
            if file["dataFile"]["filename"] in d_files
        ]
        
        if len(files_to_download) != len(d_files):
            found_names = {file["dataFile"]["filename"] for file in files_to_download}
            missing = set(d_files) - found_names
            print(f"WARNING: {len(missing)} file(s) not found in dataset: {missing}")
        
        if not files_to_download:
            print("ERROR: No matching files found.")
            return 1
        
        # Download individually
        for file in files_to_download:
            filename = file["dataFile"]["filename"]
            file_id = file["dataFile"]["id"]
            print(f"  Downloading: {filename} (ID: {file_id})")
            
            try:
                response = data_api.get_datafile(file_id, is_pid=False)
                output_path = output_dir / filename
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                print(f"    ✓ Saved to: {output_path}")
            except Exception as e:
                print(f"    ERROR: {e}")
    
    print()
    print("✓ Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

