import sys
from pathlib import Path

from dotenv import load_dotenv

# Let tests import the package as `src.agent` without installing anything.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Only the integration test needs a key. The unit tests pass a fake client and
# never read the environment, so this is a no-op for them.
load_dotenv()
