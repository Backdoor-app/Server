#!/usr/bin/env python3
"""
Script to remove all authentication requirements from the HTML template in app.py
"""

import re
import sys

def remove_auth_sections(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace all auth-info sections with <!-- No authentication required -->
    pattern = r'<div class="auth-info">\s+<strong>Authentication Required:</strong>.*?</div>'
    replacement = '<!-- No authentication required -->'
    modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(modified_content)
    
    print(f"Replaced all authentication sections in {file_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "app.py"
    
    remove_auth_sections(file_path)
