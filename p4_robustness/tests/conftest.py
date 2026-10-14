"""Let tests import p4_robustness modules (data_prep, proxy_model) by name."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
