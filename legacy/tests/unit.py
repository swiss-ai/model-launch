import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add serving directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serving"))

from utils import fetch_bootstrap_addresses


class TestFetchBootstrapAddresses(unittest.TestCase):
    """Unit tests for fetch_bootstrap_addresses function."""

    @patch("utils.urlopen")
    def test_bootstraps_format(self, mock_urlopen):
        """Test parsing /v1/dnt/bootstraps format."""
        # Load the example payload from http://148.187.108.172:8092/v1/dnt/bootstraps
        test_dir = os.path.dirname(os.path.abspath(__file__))
        payload_path = os.path.join(test_dir, "bootstrap_payload.json")

        with open(payload_path) as f:
            payload = json.load(f)

        # Mock urlopen to return our payload
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(payload).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_bootstrap_addresses()

        # Should return the first bootstrap address from the list
        self.assertEqual(result, "/ip4/148.187.108.172/tcp/43905/p2p/QmPf4rfgfHTVy6geMJX9iTmmbp4jxDpe2uWuJsTmcENwAq")


if __name__ == "__main__":
    unittest.main()
