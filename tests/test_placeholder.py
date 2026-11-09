"""Initial smoke tests to verify project scaffolding and module structure."""

import unittest

import src.answer
import src.chunk
import src.cite
import src.fuse
import src.index
import src.ingest
import src.retrieve


class TestProjectScaffolding(unittest.TestCase):
    """Test that all core pipeline modules are importable and have docstrings."""

    def test_modules_importable(self):
        """Verify that all core pipeline modules exist and can be imported."""
        self.assertIsNotNone(src.ingest.__doc__)
        self.assertIsNotNone(src.chunk.__doc__)
        self.assertIsNotNone(src.index.__doc__)
        self.assertIsNotNone(src.retrieve.__doc__)
        self.assertIsNotNone(src.fuse.__doc__)
        self.assertIsNotNone(src.answer.__doc__)
        self.assertIsNotNone(src.cite.__doc__)


if __name__ == "__main__":
    unittest.main()
