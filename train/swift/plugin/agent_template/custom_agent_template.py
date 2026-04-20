# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import TYPE_CHECKING, List, Tuple, Union

import json

from swift.plugin.agent_template.base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt


class CustomDirectToolAgentTemplate(BaseAgentTemplate):
    """
    自定义Agent模板，支持直接的tool角色对话流程：
    system -> user -> assistant(工具调用) -> tool -> assistant(最终回答)
    """

    def get_toolcall(self, response: str) -> List['Function']:
        """从assistant响应中提取工具调用"""
        from swift.llm.infer import Function
        res_list = re.findall(r'<tool_call>(.+?)</tool_call>', response, re.DOTALL)
        functions = []
        for res in res_list:
            res = self._parse_json(res)
            if isinstance(res, dict) and 'name' in res and 'arguments' in res:
                functions.append(Function(name=res['name'], arguments=res['arguments']))
        return functions

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        """格式化工具响应：将tool消息转换为user消息格式"""
        # 🔑 关键修复：将tool消息格式化为user模板，绕过角色验证
        if hasattr(self, 'template_meta'):
            prompt = self.template_meta.prompt
            chat_sep = self.template_meta.chat_sep
        else:
            # 默认使用简单的user模板格式
            prompt = ['{{QUERY}}']
            chat_sep = ['']

        res = chat_sep.copy() if chat_sep else []

        # 将所有tool消息合并为一个tool响应块
        tool_contents = []
        for tool_message in tool_messages:
            tool_contents.append(f"Tool返回结果: {tool_message['content']}")

        total_tool_response = '\n'.join(tool_contents)

        # 将tool响应替换到user模板中
        for context in prompt:
            if isinstance(context, str):
                context = context.replace('{{QUERY}}', total_tool_response)
            res.append(context)

        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: str, user_message=None) -> str:
        """直接返回system消息内容"""
        return system
