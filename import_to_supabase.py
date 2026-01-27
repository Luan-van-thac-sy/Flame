"""
Script to import MIMIC-III CSV files to Supabase.

Setup:
1. Install dependencies: pip install supabase pandas python-dotenv
2. Create .env file with:
   SUPABASE_URL=your-project-url
   SUPABASE_KEY=your-anon-key
3. Ensure tables are created in Supabase with correct schema
4. Run: python import_to_supabase.py
"""

import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Supabase Free Tier limit: 500 MB
SUPABASE_FREE_TIER_LIMIT_MB = 500

# Initialize Supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY in your .env file")

supabase: Client = create_client(url, key)

# CSV files to import
csv_files = {
    'admissions': 'data/data_process/input/mimic-iii/ADMISSIONS.csv',
    'diagnoses_icd': 'data/data_process/input/mimic-iii/DIAGNOSES_ICD.csv',
    'procedures_icd': 'data/data_process/input/mimic-iii/PROCEDURES_ICD.csv',
    'patients': 'data/data_process/input/mimic-iii/PATIENTS.csv',
    'prescriptions': 'data/data_process/input/mimic-iii/PRESCRIPTIONS.csv',
    'noteevents': 'data/data_process/input/mimic-iii/NOTEEVENTS.csv',
}

# Batch size for inserting data (adjust based on your needs and Supabase limits)
BATCH_SIZE = 1000


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    if not os.path.exists(file_path):
        return 0
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def check_storage_requirements():
    """Check total file sizes and warn about Supabase free tier limits."""
    print("\n" + "="*60)
    print("KIỂM TRA DUNG LƯỢNG FILE / CHECKING FILE SIZES")
    print("="*60)

    total_size_mb = 0
    file_sizes = {}

    for table_name, csv_path in csv_files.items():
        size_mb = get_file_size_mb(csv_path)
        file_sizes[table_name] = size_mb
        total_size_mb += size_mb

        if os.path.exists(csv_path):
            status = "✓"
        else:
            status = "✗"

        print(f"{status} {table_name:20s}: {size_mb:8.2f} MB")

    print("-"*60)
    print(f"TỔNG CỘNG / TOTAL: {total_size_mb:.2f} MB")
    print("="*60)

    # Warning about free tier
    if total_size_mb > SUPABASE_FREE_TIER_LIMIT_MB:
        print("\n⚠️  CẢNH BÁO / WARNING:")
        print(f"   Tổng dung lượng ({total_size_mb:.2f} MB) VƯỢT QUÁ giới hạn")
        print(f"   Supabase Free Tier ({SUPABASE_FREE_TIER_LIMIT_MB} MB)")
        print("\n   Giải pháp / Solutions:")
        print("   1. Nâng cấp lên Supabase Pro ($25/tháng, 8 GB)")
        print("   2. Chỉ import một phần dữ liệu (sample/subset)")
        print("   3. Sử dụng PostgreSQL local hoặc cloud database khác")
        print("\n   Bạn có muốn tiếp tục? (y/n) / Continue? (y/n): ", end="")

        response = input().strip().lower()
        if response != 'y':
            print("Đã hủy import / Import cancelled.")
            return False
    else:
        print(f"\n✓ Dung lượng ({total_size_mb:.2f} MB) nằm trong giới hạn Free Tier")

    return True


def import_csv_to_supabase(table_name: str, csv_path: str):
    """Import CSV file to Supabase table in batches."""
    print(f"\n{'='*60}")
    print(f"Importing {csv_path} to {table_name}...")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"⚠ File not found: {csv_path}")
        return

    # Read CSV in chunks
    chunk_iter = pd.read_csv(csv_path, chunksize=BATCH_SIZE, low_memory=False)

    total_rows = 0
    batch_num = 0

    for chunk in chunk_iter:
        batch_num += 1

        # Convert DataFrame to list of dictionaries
        # Replace NaN with None for proper NULL handling
        records = chunk.where(pd.notnull(chunk), None).to_dict('records')

        # Insert batch
        try:
            supabase.table(table_name).insert(records).execute()
            total_rows += len(records)
            print(f"  Batch {batch_num}: Inserted {len(records)} rows (Total: {total_rows})")
        except Exception as e:
            print(f"  ⚠ Error inserting batch {batch_num}: {e}")
            # Continue with next batch

    print(f"✓ Completed importing {table_name}: {total_rows} total rows\n")


def main():
    """Main function to import all CSV files."""
    print("Starting CSV import to Supabase...")
    print(f"Supabase URL: {url}")

    # Check storage requirements first
    if not check_storage_requirements():
        return

    # Import all CSV files
    for table_name, csv_path in csv_files.items():
        import_csv_to_supabase(table_name, csv_path)

    print("="*60)
    print("All imports completed!")
    print("="*60)


if __name__ == "__main__":
    main()
