#!/usr/bin/env python3
"""
Ollama client with SSH tunneling support
Connects to remote Ollama server via SSH tunnel and sends requests
"""

import sys
import json
import os
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

try:
    from sshtunnel import SSHTunnelForwarder
except ImportError:
    print("ERROR: sshtunnel not installed. Run: pip install sshtunnel", file=sys.stderr)
    SSHTunnelForwarder = None

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "ollama_config.json"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to config file (uses default if not provided)
    
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Please create config file with SSH and Ollama settings."
        )
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Expand ~ in key_file path
    if 'ssh' in config and 'key_file' in config['ssh']:
        config['ssh']['key_file'] = os.path.expanduser(config['ssh']['key_file'])
    
    return config


@contextmanager
def ssh_tunnel(config: Dict[str, Any]):
    """
    Create SSH tunnel to remote Ollama server
    
    Args:
        config: Configuration dictionary with SSH settings
    
    Yields:
        Local port number for the tunnel
    """
    if SSHTunnelForwarder is None:
        raise ImportError("sshtunnel library is required. Install with: pip install sshtunnel")
    
    ssh_config = config['ssh']
    ollama_config = config['ollama']
    
    # Prepare SSH connection parameters
    ssh_params = {
        'ssh_address_or_host': (ssh_config['host'], ssh_config.get('port', 22)),
        'ssh_username': ssh_config['username'],
        'remote_bind_address': (ollama_config.get('remote_host', 'localhost'), 
                                ollama_config.get('remote_port', 11434)),
        'local_bind_address': ('127.0.0.1', ollama_config.get('local_port', 0)),  # 0 = auto-assign
    }
    
    # Authentication: key file or password
    key_file = ssh_config.get('key_file')
    passphrase = ssh_config.get('passphrase')
    password = ssh_config.get('password')
    
    if key_file and os.path.exists(key_file):
        ssh_params['ssh_pkey'] = key_file
        if passphrase:
            ssh_params['ssh_private_key_password'] = passphrase
    elif password:
        ssh_params['ssh_password'] = password
    else:
        raise ValueError("SSH authentication required: provide key_file or password in config")
    
    # Create and start tunnel
    tunnel = SSHTunnelForwarder(**ssh_params)
    
    try:
        print(f"OLLAMA: Opening SSH tunnel to {ssh_config['host']}...", file=sys.stderr)
        tunnel.start()
        local_port = tunnel.local_bind_port
        print(f"OLLAMA: SSH tunnel established on local port {local_port}", file=sys.stderr)
        yield local_port
    finally:
        print("OLLAMA: Closing SSH tunnel...", file=sys.stderr)
        tunnel.stop()


def query_ollama(
    prompt: str,
    config: Dict[str, Any],
    local_port: int,
    model: Optional[str] = None,
    stream: bool = False
) -> str:
    """
    Send query to Ollama API
    
    Args:
        prompt: The prompt to send
        config: Configuration dictionary
        local_port: Local port of SSH tunnel
        model: Model name (uses config default if not provided)
        stream: Whether to stream the response
    
    Returns:
        Response text from Ollama
    """
    ollama_config = config.get('ollama', {})
    model_name = model or ollama_config.get('model', 'llama3')
    timeout = ollama_config.get('timeout', 60)
    
    url = f"http://127.0.0.1:{local_port}/api/generate"
    
    payload = {
        'model': model_name,
        'prompt': prompt,
        'stream': stream
    }
    
    # Add system prompt if configured (skip if null/None)
    prompt_config = config.get('prompt', {})
    system_prompt = prompt_config.get('system_prompt')
    if system_prompt:
        payload['system'] = system_prompt
    
    print(f"OLLAMA: Sending request to model '{model_name}'...", file=sys.stderr)
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '')
        
        print(f"OLLAMA: Response received ({len(response_text)} chars)", file=sys.stderr)
        return response_text
        
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama request timed out after {timeout} seconds")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Failed to connect to Ollama: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def chat_ollama(
    messages: list,
    config: Dict[str, Any],
    local_port: int,
    model: Optional[str] = None
) -> str:
    """
    Send chat messages to Ollama API (multi-turn conversation)
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        config: Configuration dictionary
        local_port: Local port of SSH tunnel
        model: Model name (uses config default if not provided)
    
    Returns:
        Response text from Ollama
    """
    ollama_config = config.get('ollama', {})
    model_name = model or ollama_config.get('model', 'llama3')
    timeout = ollama_config.get('timeout', 60)
    
    url = f"http://127.0.0.1:{local_port}/api/chat"
    
    payload = {
        'model': model_name,
        'messages': messages,
        'stream': False
    }
    
    print(f"OLLAMA: Sending chat request to model '{model_name}'...", file=sys.stderr)
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('message', {}).get('content', '')
        
        print(f"OLLAMA: Response received ({len(response_text)} chars)", file=sys.stderr)
        return response_text
        
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama request timed out after {timeout} seconds")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Failed to connect to Ollama: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def process_transcription(
    transcription: str,
    config_path: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Main function: Send Whisper transcription to Ollama via SSH tunnel
    
    Args:
        transcription: Transcribed text from Whisper
        config_path: Path to config file
        model: Model name override
    
    Returns:
        Ollama's response
    """
    if not transcription or not transcription.strip():
        return ""
    
    # Load configuration
    config = load_config(config_path)
    
    # Build prompt from template
    prompt_config = config.get('prompt', {})
    template = prompt_config.get('template', 'Transcription: {transcription}')
    prompt = template.format(transcription=transcription)
    
    # Connect via SSH tunnel and query Ollama
    with ssh_tunnel(config) as local_port:
        response = query_ollama(
            prompt=prompt,
            config=config,
            local_port=local_port,
            model=model
        )
    
    return response


def main():
    """CLI interface for testing Ollama connection"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Query Ollama via SSH tunnel",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: ollama_config.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (overrides config)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connection with a simple prompt"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        help="Prompt to send to Ollama"
    )
    
    args = parser.parse_args()
    
    if args.test:
        prompt = "Hello! Please respond with a brief greeting."
    elif args.prompt:
        prompt = args.prompt
    else:
        # Read from stdin
        prompt = sys.stdin.read().strip()
    
    if not prompt:
        print("ERROR: No prompt provided", file=sys.stderr)
        sys.exit(1)
    
    try:
        config = load_config(args.config)
        
        with ssh_tunnel(config) as local_port:
            response = query_ollama(
                prompt=prompt,
                config=config,
                local_port=local_port,
                model=args.model
            )
            
            # Output only the response to stdout
            print(response)
            
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
