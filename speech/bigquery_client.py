#!/usr/bin/env python3
"""
BigQuery client for transcription integration
Uploads transcription data and executes SQL queries on BigQuery
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Set up BigQuery credentials
BIGQUERY_PATH = Path(__file__).parent.parent / "ADS-B-Display" / "BigQuery"
API_KEY_PATH = BIGQUERY_PATH / "t-rex.json"

if API_KEY_PATH.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(API_KEY_PATH)

try:
    from google.cloud import bigquery
except ImportError:
    print("ERROR: google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery>=3.21.0", file=sys.stderr)
    bigquery = None

# Default configuration
DEFAULT_CONFIG_PATH = Path(__file__).parent / "bigquery_config.json"
DEFAULT_PROJECT = "iitp-class-team-4-473114"
DEFAULT_DATASET = "ADSB"
DEFAULT_LOCATION = "asia-northeast3"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load BigQuery configuration from JSON file
    
    Args:
        config_path: Path to config file (uses default if not provided)
    
    Returns:
        Configuration dictionary with defaults applied
    """
    config = {
        "project": DEFAULT_PROJECT,
        "dataset": DEFAULT_DATASET,
        "location": DEFAULT_LOCATION,
        "transcription_table": "Transcriptions",
        "query_timeout": 60
    }
    
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            print(f"BIGQUERY: Warning - Could not load config: {e}", file=sys.stderr)
    
    return config


def get_client(config: Dict[str, Any] = None) -> 'bigquery.Client':
    """
    Initialize and return BigQuery client
    
    Args:
        config: Configuration dictionary
    
    Returns:
        BigQuery client instance
    """
    if bigquery is None:
        raise ImportError("google-cloud-bigquery is not installed")
    
    if config is None:
        config = load_config()
    
    project = config.get("project", DEFAULT_PROJECT)
    location = config.get("location", DEFAULT_LOCATION)
    
    return bigquery.Client(project=project, location=location)


def format_query_results(rows: List, max_rows: int = 20) -> str:
    """
    Format BigQuery query results as a table
    
    Args:
        rows: List of Row objects from BigQuery query result
        max_rows: Maximum number of rows to display
    
    Returns:
        Formatted table string
    """
    if not rows:
        return "No rows returned."
    
    # Get column names from first row
    first_row = rows[0]
    columns = list(first_row.keys())
    
    # Calculate column widths
    col_widths = {}
    for col in columns:
        col_widths[col] = len(str(col))
        for row in rows[:max_rows]:
            value = str(row.get(col, ""))
            col_widths[col] = max(col_widths[col], min(len(value), 50))  # Cap at 50 chars
    
    # Build table
    lines = []
    
    # Header
    header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for row in rows[:max_rows]:
        row_values = []
        for col in columns:
            val = str(row.get(col, ""))
            if len(val) > 50:
                val = val[:47] + "..."
            row_values.append(val.ljust(col_widths[col]))
        lines.append(" | ".join(row_values))
    
    if len(rows) > max_rows:
        lines.append(f"\n... ({len(rows) - max_rows} more rows not shown)")
    
    return "\n".join(lines)


def execute_query(
    sql: str,
    config: Dict[str, Any] = None,
    max_rows: int = 20,
    client: 'bigquery.Client' = None
) -> Tuple[List, Dict[str, Any]]:
    """
    Execute a SQL query on BigQuery
    
    Args:
        sql: SQL query string
        config: Configuration dictionary
        max_rows: Maximum rows to return
        client: Optional existing BigQuery client
    
    Returns:
        Tuple of (rows list, job info dict)
    """
    if config is None:
        config = load_config()
    
    if client is None:
        client = get_client(config)
    
    # Clean query - remove trailing semicolons
    sql = sql.strip().rstrip(';')
    
    print(f"BIGQUERY: Executing query...", file=sys.stderr)
    print(f"BIGQUERY: SQL: {sql[:200]}{'...' if len(sql) > 200 else ''}", file=sys.stderr)
    
    try:
        query_job = client.query(sql)
        results = query_job.result()
        
        # Collect rows
        all_rows = list(results)
        
        job_info = {
            "job_id": query_job.job_id,
            "state": query_job.state,
            "total_rows": len(all_rows),
            "bytes_processed": getattr(query_job, 'total_bytes_processed', None),
            "bytes_billed": getattr(query_job, 'total_bytes_billed', None),
        }
        
        if hasattr(query_job, 'num_dml_affected_rows') and query_job.num_dml_affected_rows is not None:
            job_info["rows_affected"] = query_job.num_dml_affected_rows
        
        print(f"BIGQUERY: Query completed. Job ID: {query_job.job_id}", file=sys.stderr)
        print(f"BIGQUERY: Total rows: {len(all_rows)}", file=sys.stderr)
        
        return all_rows[:max_rows], job_info
        
    except Exception as e:
        print(f"BIGQUERY ERROR: Query failed: {e}", file=sys.stderr)
        raise


def upload_transcription(
    transcription: str,
    audio_file: Optional[str] = None,
    language: Optional[str] = None,
    config: Dict[str, Any] = None,
    client: 'bigquery.Client' = None
) -> Dict[str, Any]:
    """
    Upload transcription result to BigQuery
    
    Args:
        transcription: Transcribed text
        audio_file: Original audio file path
        language: Detected/used language
        config: Configuration dictionary
        client: Optional existing BigQuery client
    
    Returns:
        Job info dictionary
    """
    if config is None:
        config = load_config()
    
    if client is None:
        client = get_client(config)
    
    project = config.get("project", DEFAULT_PROJECT)
    dataset = config.get("dataset", DEFAULT_DATASET)
    table = config.get("transcription_table", "Transcriptions")
    
    table_id = f"{project}.{dataset}.{table}"
    
    # Prepare row data
    row = {
        "transcription": transcription,
        "audio_file": audio_file or "",
        "language": language or "unknown",
        "created_at": datetime.utcnow().isoformat()
    }
    
    print(f"BIGQUERY: Uploading transcription to {table_id}...", file=sys.stderr)
    
    try:
        errors = client.insert_rows_json(table_id, [row])
        
        if errors:
            print(f"BIGQUERY ERROR: Insert failed: {errors}", file=sys.stderr)
            raise RuntimeError(f"BigQuery insert failed: {errors}")
        
        print(f"BIGQUERY: Transcription uploaded successfully", file=sys.stderr)
        
        return {
            "success": True,
            "table": table_id,
            "row": row
        }
        
    except Exception as e:
        print(f"BIGQUERY ERROR: Upload failed: {e}", file=sys.stderr)
        raise


def execute_sql_from_transcription(
    sql: str,
    config: Dict[str, Any] = None,
    max_rows: int = 20
) -> str:
    """
    Execute SQL query generated from transcription and return formatted results
    
    This is the main function to be called after Ollama generates SQL from transcription.
    
    Args:
        sql: SQL query string (from Ollama)
        config: Configuration dictionary
        max_rows: Maximum rows to display
    
    Returns:
        Formatted result string
    """
    if not sql or not sql.strip():
        return "No SQL query provided."
    
    # Clean the SQL - remove markdown code fences if present
    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.split('\n')
        sql_lines = []
        in_code = False
        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code or not line.startswith("```"):
                sql_lines.append(line)
        sql = '\n'.join(sql_lines).strip()
    
    if config is None:
        config = load_config()
    
    try:
        rows, job_info = execute_query(sql, config, max_rows)
        
        result_lines = []
        result_lines.append(f"Query executed successfully (Job ID: {job_info['job_id']})")
        
        if job_info.get('rows_affected') is not None:
            result_lines.append(f"Rows affected: {job_info['rows_affected']}")
        
        result_lines.append("")
        result_lines.append(format_query_results(rows, max_rows))
        
        return '\n'.join(result_lines)
        
    except Exception as e:
        return f"Query execution failed: {e}"


def process_with_bigquery(
    transcription: str,
    sql_query: str,
    config_path: Optional[str] = None,
    save_transcription: bool = False,
    audio_file: Optional[str] = None,
    language: Optional[str] = None,
    max_rows: int = 20
) -> str:
    """
    Main integration function: Process transcription with BigQuery
    
    1. Optionally save transcription to BigQuery
    2. Execute SQL query (generated from transcription by Ollama)
    3. Return formatted results
    
    Args:
        transcription: Original transcription text
        sql_query: SQL query generated by Ollama
        config_path: Path to config file
        save_transcription: Whether to save transcription to BigQuery
        audio_file: Original audio file path
        language: Detected language
        max_rows: Maximum rows to display
    
    Returns:
        Formatted result string
    """
    config = load_config(config_path)
    client = get_client(config)
    
    results = []
    
    # Save transcription if requested
    if save_transcription and transcription:
        try:
            upload_result = upload_transcription(
                transcription=transcription,
                audio_file=audio_file,
                language=language,
                config=config,
                client=client
            )
            results.append(f"Transcription saved to {upload_result['table']}")
        except Exception as e:
            results.append(f"Warning: Failed to save transcription: {e}")
    
    # Execute SQL query
    if sql_query and sql_query.strip():
        query_result = execute_sql_from_transcription(sql_query, config, max_rows)
        results.append(query_result)
    
    return '\n\n'.join(results)


def main():
    """CLI interface for BigQuery operations"""
    parser = argparse.ArgumentParser(
        description="BigQuery client for transcription integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: bigquery_config.json)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Execute SQL query")
    query_parser.add_argument("sql", type=str, nargs="?", help="SQL query (or read from stdin)")
    query_parser.add_argument("--max-rows", type=int, default=20, help="Max rows to display")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload transcription")
    upload_parser.add_argument("text", type=str, nargs="?", help="Transcription text (or read from stdin)")
    upload_parser.add_argument("--audio-file", type=str, help="Original audio file path")
    upload_parser.add_argument("--language", type=str, help="Detected language")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test BigQuery connection")
    
    args = parser.parse_args()
    
    if args.command == "query":
        sql = args.sql or sys.stdin.read().strip()
        if not sql:
            print("ERROR: No SQL query provided", file=sys.stderr)
            sys.exit(1)
        
        result = execute_sql_from_transcription(sql, load_config(args.config), args.max_rows)
        print(result)
        
    elif args.command == "upload":
        text = args.text or sys.stdin.read().strip()
        if not text:
            print("ERROR: No transcription text provided", file=sys.stderr)
            sys.exit(1)
        
        try:
            result = upload_transcription(
                transcription=text,
                audio_file=args.audio_file,
                language=args.language,
                config=load_config(args.config)
            )
            print(f"Upload successful: {result}")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif args.command == "test":
        try:
            config = load_config(args.config)
            client = get_client(config)
            
            # Test query
            project = config.get("project", DEFAULT_PROJECT)
            test_sql = f"SELECT 1 as test_value"
            
            rows, job_info = execute_query(test_sql, config, client=client)
            
            print(f"BigQuery connection successful!")
            print(f"Project: {project}")
            print(f"Location: {config.get('location', DEFAULT_LOCATION)}")
            print(f"Job ID: {job_info['job_id']}")
            
        except Exception as e:
            print(f"ERROR: Connection test failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


