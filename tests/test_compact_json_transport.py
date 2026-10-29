"""紧凑约束回归：同一 Schema 必须转换为无任意空白的语法，不依赖请求级开关。"""

from copy import deepcopy
import importlib
from pathlib import Path
import sys
import types
import unittest

package = '_compact_transport_test_core'
module = types.ModuleType(package)
module.__path__ = [str(Path(__file__).resolve().parents[1] / 'coregraph')]
sys.modules[package] = module
api = importlib.import_module(package + '.evidence_support.transport')


class CompactTransportTests(unittest.TestCase):
    def test_compiles_same_schema_without_whitespace_and_preserves_options(self):
        schema = dict(type='object', properties={'value': {'type': 'string'}}, required=['value'], additionalProperties=False)
        options = dict(max_tokens=None, response_format=dict(type='json_schema', json_schema=dict(schema=schema)),
                       extra_body=dict(chat_template_kwargs=dict(enable_thinking=False)))
        before = deepcopy(options); captured = {}
        def compiler(actual):
            captured.update(actual)
            return 'root ::= "{}"'
        result = api.compact_json_options(options, compiler=compiler)
        self.assertEqual(captured, schema)
        self.assertEqual(options, before)
        self.assertNotIn('response_format', result)
        self.assertEqual(result['extra_body']['structured_outputs'], dict(grammar='root ::= "{}"'))
        self.assertFalse(result['extra_body']['chat_template_kwargs']['enable_thinking'])
        self.assertIsNone(result['max_tokens'])

    def test_existing_constraints_cannot_be_silently_overwritten(self):
        with self.assertRaises(ValueError):
            api.compact_json_options(dict(response_format=dict(type='json_schema', json_schema=dict(schema={})),
                extra_body=dict(structured_outputs=dict(regex='.*'))), compiler=lambda schema:'grammar')
