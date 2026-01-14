#!/usr/bin/env python3
"""
Test script for text-to-SQL without microphone
Allows testing the Ollama + BigQuery pipeline with text input instead of audio
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Import Ollama client
OLLAMA_AVAILABLE = False
try:
    from ollama_client import process_transcription, load_config as load_ollama_config
    OLLAMA_AVAILABLE = True
except ImportError:
    print("ERROR: Ollama client not available. Install: pip install sshtunnel requests", file=sys.stderr)

# Import BigQuery client
BIGQUERY_AVAILABLE = False
try:
    from bigquery_client import execute_sql_from_transcription, load_config as load_bq_config
    BIGQUERY_AVAILABLE = True
except ImportError:
    print("ERROR: BigQuery client not available. Install: pip install google-cloud-bigquery>=3.21.0", file=sys.stderr)


def test_text_to_sql(
    text: str,
    ollama_config_path: Optional[str] = None,
    bigquery_config_path: Optional[str] = None,
    ollama_model: Optional[str] = None,
    max_rows: int = 20,
    show_sql: bool = True
):
    """
    Test the text-to-SQL pipeline without microphone
    
    Args:
        text: Natural language query text
        ollama_config_path: Path to Ollama config
        bigquery_config_path: Path to BigQuery config
        ollama_model: Override Ollama model name
        max_rows: Maximum rows to display
        show_sql: Whether to show the generated SQL
    """
    if not text or not text.strip():
        print("ERROR: No text provided", file=sys.stderr)
        return
    
    print(f"📝 Input Query: {text}\n", file=sys.stdout)
    sys.stdout.flush()
    
    # Step 1: Send to Ollama for SQL generation
    sql_query = None
    if OLLAMA_AVAILABLE:
        try:
            print("🤖 Generating SQL with Ollama...", file=sys.stderr)
            sys.stderr.flush()
            
            sql_query = process_transcription(
                transcription=text,
                config_path=ollama_config_path,
                model=ollama_model
            )
            
            if sql_query:
                if show_sql:
                    print(f"\n📊 Generated SQL:", file=sys.stdout)
                    print("-" * 60, file=sys.stdout)
                    print(sql_query, file=sys.stdout)
                    print("-" * 60, file=sys.stdout)
                    sys.stdout.flush()
                
                print("✅ SQL generated successfully", file=sys.stderr)
                sys.stderr.flush()
            else:
                print("⚠️  No SQL query generated", file=sys.stderr)
                sys.stderr.flush()
                
        except FileNotFoundError as e:
            print(f"❌ ERROR: Config file not found - {e}", file=sys.stderr)
            sys.stderr.flush()
            return
        except Exception as e:
            print(f"❌ ERROR: Ollama processing failed - {e}", file=sys.stderr)
            sys.stderr.flush()
            return
    else:
        print("❌ ERROR: Ollama client not available", file=sys.stderr)
        return
    
    # Step 2: Execute SQL on BigQuery
    if sql_query and BIGQUERY_AVAILABLE:
        try:
            print("\n🔍 Executing query on BigQuery...", file=sys.stderr)
            sys.stderr.flush()
            
            bq_config = load_bq_config(bigquery_config_path)
            bq_result = execute_sql_from_transcription(
                sql=sql_query,
                config=bq_config,
                max_rows=max_rows
            )
            
            if bq_result:
                print(f"\n📈 BigQuery Results:", file=sys.stdout)
                print("=" * 60, file=sys.stdout)
                print(bq_result, file=sys.stdout)
                print("=" * 60, file=sys.stdout)
                sys.stdout.flush()
                
                print("✅ Query executed successfully", file=sys.stderr)
                sys.stderr.flush()
            else:
                print("⚠️  No results returned", file=sys.stderr)
                sys.stderr.flush()
                
        except Exception as e:
            print(f"❌ ERROR: BigQuery execution failed - {e}", file=sys.stderr)
            sys.stderr.flush()
    elif not BIGQUERY_AVAILABLE:
        print("⚠️  BigQuery client not available - skipping execution", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Test text-to-SQL pipeline without microphone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with direct text input
  python test_with_text.py "Show me all flights with altitude above 10000"
  
  # Test from a file
  python test_with_text.py --file examples.txt
  
  # Hide SQL in output
  python test_with_text.py --no-sql "Show latest flight positions"
        """
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        help="Natural language query text (or read from stdin/file)"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Read queries from file (one per line)"
    )
    
    parser.add_argument(
        "--ollama-config",
        type=str,
        default=None,
        help="Path to Ollama config file (default: ollama_config.json)"
    )
    
    parser.add_argument(
        "--bigquery-config",
        type=str,
        default=None,
        help="Path to BigQuery config file (default: bigquery_config.json)"
    )
    
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Ollama model name (overrides config)"
    )
    
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Maximum rows to display (default: 20)"
    )
    
    parser.add_argument(
        "--no-sql",
        action="store_true",
        help="Hide generated SQL in output"
    )
    
    args = parser.parse_args()
    
    # Get input text
    queries = []
    
    if args.file:
        # Read from file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    queries.append(line)
    
    elif args.text:
        # Use command-line argument
        queries.append(args.text)
    
    else:
        # Read from stdin
        print("Enter your query (Ctrl+D or Ctrl+Z to finish):", file=sys.stderr)
        text = sys.stdin.read().strip()
        if text:
            queries.append(text)
    
    if not queries:
        print("ERROR: No query provided", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Process each query
    for i, query in enumerate(queries):
        if len(queries) > 1:
            print(f"\n{'='*60}", file=sys.stdout)
            print(f"Query {i+1} of {len(queries)}", file=sys.stdout)
            print(f"{'='*60}\n", file=sys.stdout)
        
        test_text_to_sql(
            text=query,
            ollama_config_path=args.ollama_config,
            bigquery_config_path=args.bigquery_config,
            ollama_model=args.ollama_model,
            max_rows=args.max_rows,
            show_sql=not args.no_sql
        )
        
        if len(queries) > 1 and i < len(queries) - 1:
            print("\n" + "="*60 + "\n", file=sys.stdout)


if __name__ == "__main__":
    main()



