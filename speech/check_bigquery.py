#!/usr/bin/env python3
"""
Diagnostic script to check BigQuery datasets and their locations
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up BigQuery credentials
BIGQUERY_PATH = Path(__file__).parent.parent / "ADS-B-Display" / "BigQuery"
API_KEY_PATH = BIGQUERY_PATH / "t-rex.json"

if API_KEY_PATH.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(API_KEY_PATH)
    print(f"✅ Found credentials: {API_KEY_PATH}\n")
else:
    print(f"❌ Credentials not found: {API_KEY_PATH}")
    sys.exit(1)

try:
    from google.cloud import bigquery
except ImportError:
    print("ERROR: google-cloud-bigquery not installed")
    print("Run: pip install google-cloud-bigquery>=3.21.0")
    sys.exit(1)

# Create client without specifying location (let it auto-detect)
print("📊 Checking BigQuery configuration...\n")
client = bigquery.Client()

print(f"Project ID: {client.project}")
print(f"Default Location: {client.location if client.location else 'Not set (auto-detect)'}\n")

# List all datasets
print("=" * 60)
print("Available Datasets:")
print("=" * 60)

datasets = list(client.list_datasets())

if datasets:
    for dataset in datasets:
        dataset_ref = client.get_dataset(dataset.dataset_id)
        print(f"\n📁 Dataset: {dataset.dataset_id}")
        print(f"   Full ID: {dataset.project}.{dataset.dataset_id}")
        print(f"   Location: {dataset_ref.location}")
        print(f"   Created: {dataset_ref.created}")
        
        # List tables in this dataset
        tables = list(client.list_tables(dataset.dataset_id))
        if tables:
            print(f"   Tables ({len(tables)}):")
            for table in tables[:10]:  # Show first 10 tables
                table_ref = client.get_table(f"{dataset.project}.{dataset.dataset_id}.{table.table_id}")
                print(f"      - {table.table_id} ({table_ref.num_rows:,} rows)")
            if len(tables) > 10:
                print(f"      ... and {len(tables) - 10} more tables")
        else:
            print(f"   Tables: (none)")
else:
    print("\n⚠️  No datasets found in this project")

# Try to query the specific dataset
print("\n" + "=" * 60)
print("Testing ADSB Dataset:")
print("=" * 60)

try:
    dataset_ref = client.get_dataset("ADSB")
    print(f"✅ Dataset 'ADSB' found!")
    print(f"   Full ID: {dataset_ref.project}.{dataset_ref.dataset_id}")
    print(f"   Location: {dataset_ref.location}")
    
    # Check for Flight table
    try:
        table_ref = client.get_table("ADSB.Flight")
        print(f"\n✅ Table 'Flight' found!")
        print(f"   Full ID: {table_ref.full_table_id}")
        print(f"   Rows: {table_ref.num_rows:,}")
        print(f"   Size: {table_ref.num_bytes:,} bytes")
        
        # Try a simple query
        print(f"\n🔍 Testing query...")
        query = "SELECT COUNT(*) as total_rows FROM `iitp-class-team-4-473114.ADSB.Flight` LIMIT 1"
        query_job = client.query(query)
        results = query_job.result()
        for row in results:
            print(f"   Total rows in Flight table: {row.total_rows:,}")
        
    except Exception as e:
        print(f"❌ Table 'Flight' not found: {e}")
        
except Exception as e:
    print(f"❌ Dataset 'ADSB' not found: {e}")

print("\n" + "=" * 60)
print("💡 Recommended Configuration:")
print("=" * 60)

if datasets:
    for dataset in datasets:
        if dataset.dataset_id == "ADSB":
            dataset_ref = client.get_dataset(dataset.dataset_id)
            print(f"""
Update your bigquery_config.json with:

{{
    "project": "{client.project}",
    "dataset": "ADSB",
    "location": "{dataset_ref.location}",
    "transcription_table": "Transcriptions",
    "query_timeout": 60
}}
""")
            break
else:
    print("\n⚠️  Could not determine configuration - no datasets found")

print("=" * 60)

