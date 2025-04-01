import sys
import os

project_home = u'/home/yourusername/backdoor_ai'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

from app import app as application