"""模型传输回归：显式取消输出预算后，请求不能重新带上默认 max_tokens。"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class OutputBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, response_content='ok', **options):
        """加载实际传输方法，固定客户端仅记录参数；不调用模型、不安装依赖。"""
        path = Path(__file__).resolve().parents[1] / 'base/llm.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                      and n.name == '_generate_internal')
        namespace = dict(count_input_tokens=lambda *a: 1, count_output_tokens=lambda *a: 1,
                         logger=SimpleNamespace(error=lambda *a: None))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        captured = {}
        class Client:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response_content),
                                                               finish_reason='stop')], usage=None)
        client = Client()
        manager = SimpleNamespace(api_key='EMPTY', model='test', enable_thinking=False,
            client=SimpleNamespace(chat=SimpleNamespace(completions=client)),
            cost_manager=SimpleNamespace(update_cost=lambda *a: None))
        await namespace['_generate_internal'](manager, 'test', **options)
        return captured

    async def test_explicit_none_omits_output_token_parameter(self):
        request = await self.request(max_tokens=None)
        self.assertNotIn('max_tokens', request)
        self.assertNotIn('max_completion_tokens', request)

    async def test_existing_default_and_explicit_limits_remain_compatible(self):
        self.assertEqual((await self.request())['max_tokens'], 1000)
        self.assertEqual((await self.request(max_tokens=300))['max_tokens'], 300)

    async def test_audit_keeps_raw_whitespace_before_strip(self):
        metadata = {}
        raw = '{"status":"ok"}' + '\r\n  ' * 100
        await self.request(response_content=raw, metadata_sink=metadata)
        self.assertEqual(metadata['raw_response_content'], raw)
        self.assertEqual(metadata['trailing_whitespace_chars'], 400)


if __name__ == '__main__':
    unittest.main()
