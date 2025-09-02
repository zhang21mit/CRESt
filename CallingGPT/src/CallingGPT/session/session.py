import json
import logging
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessageToolCall
from ..entities.namespace import Namespace


class Session:
    namespace: Namespace = None

    messages: list = []

    model: str = "gpt-4-1106-preview"

    def __init__(
            self,
            modules: list,
            model: str = model,
            system_prompt: str = "",
            **kwargs
    ):
        self.client = OpenAI()
        self.namespace = Namespace(modules)
        self.model = model
        self.messages.append(
            {
                "role": "system",
                "content": system_prompt
            }
        )
        self.resp_log = []

        self.args = {
            "model": self.model,
            "messages": self.messages,
            **kwargs
        }
        if len(self.namespace.functions_list) > 0:
            self.args['tools'] = []
            for function_tool in self.namespace.functions_list:
                self.args['tools'].append({
                    "type": "function",
                    "function": function_tool,
                })
            self.args['tool_choice'] = "auto"

    def ask(self, msg: str, fc_chain: bool = True) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": msg
            }
        )

        self.print_gpt_process(msg, 'msg_received')

        resp = self.client.chat.completions.create(
            **self.args
        )
        self.resp_log.append(resp)

        logging.debug("Response: {}".format(resp))
        reply_msg = resp.choices[0].message
        self.messages.append(reply_msg)

        if fc_chain:
            while reply_msg.tool_calls and any([tc.type == 'function' for tc in reply_msg.tool_calls]):
                resp = self.fc_chain(reply_msg.tool_calls)
                reply_msg = resp.choices[0].message
            ret = {
                "type": "message",
                "value": reply_msg.content,
            }

            self.messages.append({
                "role": "assistant",
                "content": reply_msg.content
            })

            self.print_gpt_process(reply_msg.content, 'msg_replied')

            return ret['value']

        else:
            if 'function_call' in reply_msg:

                fc = reply_msg.function_call
                args = json.loads(fc.arguments)
                call_ret = self._call_function(fc.name, args)

                self.messages.append({
                    "role": "function",
                    "name": fc.name,
                    "content": str(call_ret)
                })

                ret = {
                    "type": "function_call",
                    "func": fc.name.replace('-', '.'),
                    "value": call_ret,
                }
            else:
                ret = {
                    "type": "message",
                    "value": reply_msg.content,
                }

                self.messages.append({
                    "role": "assistant",
                    "content": reply_msg.content
                })

            return ret['value']

    def fc_chain(self, fc_cmd_list: list[ChatCompletionMessageToolCall]) -> ChatCompletion:
        """
        Excecute the function call and return the result to ChatGPT.

        Args:
            fc_cmd_list (list[ChatCompletionMessageToolCall]): The function call list from ChatGPT.

        Returns:
            dict: The response from ChatGPT.
        """
        for fc_cmd in fc_cmd_list:
            if fc_cmd.type == 'function':
                content = (f"function name:\n"
                           f"{fc_cmd.function.name}\n\n"
                           f"args:\n"
                           f"{fc_cmd.function.arguments}")
                self.print_gpt_process(content, 'func_called')
                fc_args = json.loads(fc_cmd.function.arguments)

                call_ret = self._call_function(fc_cmd.function.name, fc_args)
                self.print_gpt_process(f"ret_value:\n{call_ret}", 'func_completed')

                self.messages.append({
                    "tool_call_id": fc_cmd.id,
                    "role": "tool",
                    "name": fc_cmd.function.name,
                    "content": str(call_ret)
                })
        resp = self.client.chat.completions.create(
            **self.args
        )
        self.messages.append(resp.choices[0].message)
        self.resp_log.append(resp)

        return resp

    def _call_function(self, function_name: str, args: dict):
        return self.namespace.call_function(function_name, args)

    @staticmethod
    def print_gpt_process(content, gpt_process):
        assert gpt_process in ['msg_received', 'msg_replied', 'func_called', 'func_completed']
        mapping = {
            'msg_received': 'MESSAGE RECEIVED',
            'msg_replied': 'MESSAGE REPLIED',
            'func_called': 'FUNCTION CALLED',
            'func_completed': 'FUNCTION COMPLETED',
        }
        print(f"****{mapping[gpt_process]}****\n\n"
              f"{content}\n\n\n")


class GPT_Session_Handler:
    """
    Singleton class to handle ChatSession object
    """
    _gpt_session = None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._gpt_session is None:
            cls._gpt_session = Session(*args, **kwargs)
        return cls._gpt_session
