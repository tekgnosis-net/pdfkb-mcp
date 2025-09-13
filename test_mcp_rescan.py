#!/usr/bin/env python3
"""
Test script for the new rescan_documents MCP command.
"""
import json

import requests


def test_mcp_rescan():
    """Test the rescan_documents MCP command via HTTP transport."""

    # MCP server endpoint
    mcp_url = "http://localhost:8000/mcp/call"

    # MCP request payload
    payload = {"method": "tools/call", "params": {"name": "rescan_documents", "arguments": {}}}

    headers = {"Content-Type": "application/json"}

    try:
        print("🔍 Testing MCP rescan_documents command...")
        print(f"📡 Sending request to: {mcp_url}")
        print(f"📨 Payload: {json.dumps(payload, indent=2)}")

        response = requests.post(mcp_url, json=payload, headers=headers, timeout=30)

        print(f"📊 Response Status: {response.status_code}")
        print(f"📥 Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📝 Response Text: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"📝 Raw response: {response.text}")


if __name__ == "__main__":
    test_mcp_rescan()
