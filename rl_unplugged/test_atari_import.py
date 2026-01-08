# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test script to verify the atari module can be imported without gin-config errors.

This test verifies the fix for GitHub issue #428:
https://github.com/google-deepmind/deepmind-research/issues/428

The issue was that newer versions of gin-config renamed the 'blacklist' parameter
to 'denylist' in gin.configurable(), causing a TypeError when importing the atari
module (which depends on dopamine-rl, which uses gin-config).

The fix is to pin gin-config==0.3.0 which still supports the 'blacklist' parameter.
"""

import sys
import unittest


class TestAtariImport(unittest.TestCase):
    """Test that the atari module can be imported without errors."""

    def test_import_atari_module(self):
        """Test that importing rl_unplugged.atari does not raise TypeError.
        
        This test verifies that the gin-config blacklist/denylist issue is fixed.
        The error that was occurring:
            TypeError: configurable() got an unexpected keyword argument 'blacklist'
        """
        try:
            # This import chain triggers the gin-config issue:
            # atari.py -> dopamine.discrete_domains.atari_lib -> gin.configurable
            from rl_unplugged import atari
            # If we get here without TypeError, the fix is working
            self.assertTrue(True)
        except ImportError as e:
            # Dependencies not installed - skip the test
            self.skipTest(f"Dependencies not installed: {e}")
        except TypeError as e:
            if "blacklist" in str(e) or "denylist" in str(e):
                self.fail(
                    f"gin-config compatibility issue not fixed: {e}\n"
                    "Make sure gin-config==0.3.0 is installed."
                )
            raise

    def test_atari_module_has_expected_attributes(self):
        """Test that the atari module has expected functions after import."""
        try:
            from rl_unplugged import atari
            
            # Check that key functions exist
            self.assertTrue(hasattr(atari, 'dataset'))
            self.assertTrue(hasattr(atari, 'environment'))
            self.assertTrue(hasattr(atari, 'AtariDopamineWrapper'))
            self.assertTrue(hasattr(atari, 'TUNING_SUITE'))
            self.assertTrue(hasattr(atari, 'TESTING_SUITE'))
            self.assertTrue(hasattr(atari, 'ALL'))
        except ImportError as e:
            self.skipTest(f"Dependencies not installed: {e}")
        except TypeError as e:
            if "blacklist" in str(e) or "denylist" in str(e):
                self.fail(
                    f"gin-config compatibility issue not fixed: {e}\n"
                    "Make sure gin-config==0.3.0 is installed."
                )
            raise


if __name__ == '__main__':
    unittest.main()
