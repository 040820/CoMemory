"""证据模块的模型传输与审计：将相同 JSON Schema 编译成真正紧凑的语法。

为什么需要这个适配层：
    vLLM 当前版本的请求级 disable_any_whitespace 不能替代引擎配置，模型可能在
    JSON 的 relation 字段后持续生成空白。本适配器把同一份完整 Schema 编译成
    无自由空白的 EBNF，直接放进 structured_outputs.grammar；不重启服务、
    不修改判断字段、不恢复 Token 上限。

审计边界：
    每次请求都保存原始响应、finish_reason、实际使用的语法和 Schema；
    返回给调用方的字符串仍可被上层 strip，但审计文件保留原始内容。
"""

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path


@lru_cache(maxsize=64)
def _compile_schema(canonical_schema):
    """CPU 语法编译并缓存；缓存键包含完整 Schema，字段变化不会复用旧语法。

    canonical_schema 是 sort_keys=True 的规范 JSON 字符串，保证同一 Schema
    的不同构造顺序命中同一缓存；不同 Schema 不会共用规则。
    """
    try:
        import xgrammar
    except ImportError as error:
        raise RuntimeError('紧凑解码需要 xgrammar==0.2.7；请安装 requirements/evidence_support.txt') from error
    # any_whitespace=False + separators=(',', ':')：只允许规范 JSON 形式，
    # 不改变字段、枚举或必填约束。
    return str(xgrammar.Grammar.from_json_schema(canonical_schema, any_whitespace=False,
                                                separators=(',', ':')))


def compact_json_options(options, *, compiler=None):
    """服务于 vLLM 请求：保留同一 Schema 的全部约束，仅替换输出语法形式。

    compiler 是测试注入点，正式入口使用 xgrammar；不能静默覆盖另一组约束。
    返回副本，max_tokens=None 等选项原样保留，不改变调用方参数。
    """
    result = deepcopy(options)
    response_format = result.pop('response_format')
    if response_format.get('type') != 'json_schema':
        raise ValueError('紧凑语法适配只接受 json_schema')
    schema = response_format['json_schema']['schema']
    extra = result.setdefault('extra_body', {})
    # 已经存在生成约束时拒绝覆盖：否则调用方以为用的是 A 规则，实际却是 B。
    if any(key in extra for key in ('structured_outputs', 'guided_json', 'guided_grammar')):
        raise ValueError('请求已有生成约束，禁止静默覆盖')
    grammar = (compiler(deepcopy(schema)) if compiler is not None else
               _compile_schema(json.dumps(schema, sort_keys=True, ensure_ascii=False)))
    if not isinstance(grammar, str) or not grammar.strip():
        raise ValueError('Schema 编译结果为空')
    extra['structured_outputs'] = dict(grammar=grammar)
    return result


class AuditedLLM:
    """证据实验模型装配器：统一保存实际语法、Schema、请求与未经裁剪的响应元数据。"""

    def __init__(self, manager, directory, *, compact=True):
        """包装一个已有 LLM 管理器，并把审计文件写到 directory。

        compact=True 时只把 json_schema 请求转换成紧凑语法；其他请求保持原样。
        directory 必须由实验入口创建为独立目录，不能覆盖历史运行。
        """
        self.manager, self.directory, self.compact = manager, Path(directory), compact
        self.directory.mkdir(parents=True, exist_ok=True)

    async def generate(self, prompt, **kwargs):
        """调用底层模型并记录完整审计；失败也写入 trace 后原样抛出。"""
        # metadata_sink 由底层填入 finish_reason/token 用量；这里接管后统一写 trace。
        metadata = kwargs.pop('metadata_sink', {})
        trace = dict(prompt=prompt, requested_max_tokens=kwargs.get('max_tokens'),
                     output_token_policy='server_default' if kwargs.get('max_tokens') is None else 'explicit_limit')
        try:
            if self.compact and (kwargs.get('response_format') or {}).get('type') == 'json_schema':
                schema = kwargs['response_format']['json_schema']['schema']
                kwargs = compact_json_options(kwargs)
                grammar = kwargs['extra_body']['structured_outputs']['grammar']
                digest = sha256(grammar.encode()).hexdigest()
                trace.update(structured_output_mode='compact_grammar', grammar_sha256=digest)
                # 每个不同语法只保存一次，完整规则可复查，避免只记录一个无效布尔开关。
                grammar_file = self.directory / ('grammar_' + digest + '.ebnf')
                if not grammar_file.exists():
                    grammar_file.write_text(grammar, encoding='utf-8')
                    grammar_file.with_suffix('.schema.json').write_text(
                        json.dumps(schema, ensure_ascii=False, indent=2), encoding='utf-8')
            else:
                trace['structured_output_mode'] = 'standard'
            response = await self.manager.generate(prompt, metadata_sink=metadata, **kwargs)
            trace['raw_response'] = response
            return response
        except BaseException as error:
            # 包括 CancelledError：先记录失败原因，再原样抛出，不吞取消信号。
            trace['error'] = f'{type(error).__name__}:{error}'
            raise
        finally:
            # finally 保证成功、失败、取消都会留下一条 trace，方便核对实际请求。
            trace['metadata'] = metadata
            with (self.directory / 'llm_trace.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(trace, ensure_ascii=False, default=str) + '\n')
