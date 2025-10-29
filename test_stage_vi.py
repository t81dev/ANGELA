#!/usr/bin/env python

import unittest
import sys
sys.path.append('release/v5.1.0')
from meta_cognition import ReflectionDescriptor

class TestStageVI(unittest.TestCase):
    def test_reflection_descriptor(self):
        descriptor = ReflectionDescriptor(
            agent_id="test_agent",
            max_depth=2,
            cadence_ms=250,
            privacy_mask={"PII"},
            consent_token="test_token"
        )
        self.assertEqual(descriptor.agent_id, "test_agent")

if __name__ == '__main__':
    unittest.main()
