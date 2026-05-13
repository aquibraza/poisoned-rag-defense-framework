#!/usr/bin/env python3
"""
Fix for sentence_transformers compatibility with newer huggingface_hub versions.
This adds the missing REPO_ID_SEPARATOR constant that sentence_transformers==2.1.0 expects.
"""

import sys
import huggingface_hub
from types import ModuleType

# Create a mock snapshot_download module with REPO_ID_SEPARATOR
mock_snapshot_download = ModuleType('snapshot_download')
mock_snapshot_download.REPO_ID_SEPARATOR = '/'

# Add the actual snapshot_download function to the mock module
try:
    from huggingface_hub import snapshot_download as actual_snapshot_download
    mock_snapshot_download.snapshot_download = actual_snapshot_download
except ImportError:
    pass

# Monkey patch the import path that sentence_transformers uses
sys.modules['huggingface_hub.snapshot_download'] = mock_snapshot_download

# Also add REPO_ID_SEPARATOR to the main huggingface_hub module
if not hasattr(huggingface_hub, 'REPO_ID_SEPARATOR'):
    huggingface_hub.REPO_ID_SEPARATOR = '/'

print("✅ Fixed sentence_transformers compatibility with huggingface_hub")
