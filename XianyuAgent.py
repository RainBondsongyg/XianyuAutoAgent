import re
from typing import List, Dict
import os
from openai import OpenAI
from loguru import logger


class XianyuReplyBot:
    def __init__(self):
        # 初始化OpenAI客户端（容错：无密钥时降级为本地占位客户端）
        self.client = self._init_llm_client()
        self._init_system_prompts()
        self._init_agents()
        self.router = IntentRouter(self.agents['classify'])
        self.last_intent = None  # 记录最后一次意图


    def _init_llm_client(self):
        """初始化LLM客户端；当无法获取到密钥时，使用占位客户端避免报错中断。"""
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("MODEL_BASE_URL")

        class _DummyMessage:
            def __init__(self, content: str):
                self.content = content

        class _DummyChoice:
            def __init__(self, content: str):
                self.message = _DummyMessage(content)

        class _DummyResponse:
            def __init__(self, content: str):
                self.choices = [_DummyChoice(content)]

        class _DummyCompletions:
            def create(self, **kwargs):
                return _DummyResponse("抱歉，当前未配置模型密钥，已切换为安全降级模式的固定回复。")

        class _DummyChat:
            def __init__(self):
                self.completions = _DummyCompletions()

        class DummyClient:
            is_dummy = True
            def __init__(self):
                self.chat = _DummyChat()

        try:
            if not api_key:
                logger.warning("未检测到 API_KEY，使用占位客户端运行（不会调用外部模型）")
                return DummyClient()
            if base_url:
                return OpenAI(api_key=api_key, base_url=base_url)
            else:
                return OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"初始化LLM客户端失败，将使用占位客户端：{e}")
            return DummyClient()

    def _init_agents(self):
        """初始化各领域Agent"""
        self.agents = {
            'classify':ClassifyAgent(self.client, self.classify_prompt, self._safe_filter),
            'price': PriceAgent(self.client, self.price_prompt, self._safe_filter),
            'tech': TechAgent(self.client, self.tech_prompt, self._safe_filter),
            'default': DefaultAgent(self.client, self.default_prompt, self._safe_filter),
        }

    def _init_system_prompts(self):
        """初始化各Agent专用提示词；当文件缺失时使用内置默认内容，不中断程序。"""
        prompt_dir = "prompts"

        def _read_prompt(filename_base: str, default_content: str) -> str:
            candidates = [
                os.path.join(prompt_dir, f"{filename_base}.txt"),
                os.path.join(prompt_dir, f"{filename_base}_example.txt"),
            ]
            for path in candidates:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        logger.debug(f"已加载提示词: {path}，长度: {len(content)} 字符")
                        return content
                except Exception:
                    continue
            logger.warning(f"提示词文件缺失，使用默认内容: {filename_base}")
            return default_content

        self.classify_prompt = _read_prompt("classify_prompt", "请判断用户信息属于技术咨询、价格议价或默认闲聊中的哪一类，仅返回 tech/price/default 之一。")
        self.price_prompt = _read_prompt("price_prompt", "请围绕价格进行友好、简洁的议价回复，注意礼貌、避免敏感信息。")
        self.tech_prompt = _read_prompt("tech_prompt", "请基于商品信息进行简要技术解答，突出关键参数和适配性。")
        self.default_prompt = _read_prompt("default_prompt", "请进行友好且简短的客服式回复，避免涉及平台外联系方式。")
        logger.info("提示词加载完成（包含可能的默认内容）")

    def _safe_filter(self, text: str) -> str:
        """安全过滤模块"""
        blocked_phrases = ["微信", "QQ", "支付宝", "银行卡", "线下"]
        return "[安全提醒]请通过平台沟通" if any(p in text for p in blocked_phrases) else text

    def format_history(self, context: List[Dict]) -> str:
        """格式化对话历史，返回完整的对话记录"""
        # 过滤掉系统消息，只保留用户和助手的对话
        user_assistant_msgs = [msg for msg in context if msg['role'] in ['user', 'assistant']]
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in user_assistant_msgs])

    def generate_reply(self, user_msg: str, item_desc: str, context: List[Dict]) -> str:
        """生成回复主流程"""
        # 记录用户消息
        # logger.debug(f'用户所发消息: {user_msg}')
        
        formatted_context = self.format_history(context)
        # logger.debug(f'对话历史: {formatted_context}')
        
        # 1. 路由决策
        detected_intent = self.router.detect(user_msg, item_desc, formatted_context)



        # 2. 获取对应Agent

        internal_intents = {'classify'}  # 定义不对外开放的Agent

        if detected_intent in self.agents and detected_intent not in internal_intents:
            agent = self.agents[detected_intent]
            logger.info(f'意图识别完成: {detected_intent}')
            self.last_intent = detected_intent  # 保存当前意图
        else:
            agent = self.agents['default']
            logger.info(f'意图识别完成: default')
            self.last_intent = 'default'  # 保存当前意图
        
        # 3. 获取议价次数
        bargain_count = self._extract_bargain_count(context)
        logger.info(f'议价次数: {bargain_count}')

        # 4. 生成回复
        return agent.generate(
            user_msg=user_msg,
            item_desc=item_desc,
            context=formatted_context,
            bargain_count=bargain_count
        )
    
    def _extract_bargain_count(self, context: List[Dict]) -> int:
        """
        从上下文中提取议价次数信息
        
        Args:
            context: 对话历史
            
        Returns:
            int: 议价次数，如果没有找到则返回0
        """
        # 查找系统消息中的议价次数信息
        for msg in context:
            if msg['role'] == 'system' and '议价次数' in msg['content']:
                try:
                    # 提取议价次数
                    match = re.search(r'议价次数[:：]\s*(\d+)', msg['content'])
                    if match:
                        return int(match.group(1))
                except Exception:
                    pass
        return 0

    def reload_prompts(self):
        """重新加载所有提示词"""
        logger.info("正在重新加载提示词...")
        self._init_system_prompts()
        self._init_agents()
        logger.info("提示词重新加载完成")


class IntentRouter:
    """意图路由决策器"""

    def __init__(self, classify_agent):
        self.rules = {
            'tech': {  # 技术类优先判定
                'keywords': ['参数', '规格', '型号', '连接', '对比'],
                'patterns': [
                    r'和.+比'             
                ]
            },
            'price': {
                'keywords': ['便宜', '价', '砍价', '少点'],
                'patterns': [r'\d+元', r'能少\d+']
            }
        }
        self.classify_agent = classify_agent

    def detect(self, user_msg: str, item_desc, context) -> str:
        """三级路由策略（技术优先）"""
        text_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', user_msg)
        
        # 1. 技术类关键词优先检查
        if any(kw in text_clean for kw in self.rules['tech']['keywords']):
            # logger.debug(f"技术类关键词匹配: {[kw for kw in self.rules['tech']['keywords'] if kw in text_clean]}")
            return 'tech'
            
        # 2. 技术类正则优先检查
        for pattern in self.rules['tech']['patterns']:
            if re.search(pattern, text_clean):
                # logger.debug(f"技术类正则匹配: {pattern}")
                return 'tech'

        # 3. 价格类检查
        for intent in ['price']:
            if any(kw in text_clean for kw in self.rules[intent]['keywords']):
                # logger.debug(f"价格类关键词匹配: {[kw for kw in self.rules[intent]['keywords'] if kw in text_clean]}")
                return intent
            
            for pattern in self.rules[intent]['patterns']:
                if re.search(pattern, text_clean):
                    # logger.debug(f"价格类正则匹配: {pattern}")
                    return intent
        
        # 4. 大模型兜底
        # logger.debug("使用大模型进行意图分类")
        return self.classify_agent.generate(
            user_msg=user_msg,
            item_desc=item_desc,
            context=context
        )


class BaseAgent:
    """Agent基类"""

    def __init__(self, client, system_prompt, safety_filter):
        self.client = client
        self.system_prompt = system_prompt
        self.safety_filter = safety_filter

    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int = 0) -> str:
        """生成回复模板方法"""
        messages = self._build_messages(user_msg, item_desc, context)
        response = self._call_llm(messages)
        return self.safety_filter(response)

    def _build_messages(self, user_msg: str, item_desc: str, context: str) -> List[Dict]:
        """构建消息链"""
        return [
            {"role": "system", "content": f"【商品信息】{item_desc}\n【你与客户对话历史】{context}\n{self.system_prompt}"},
            {"role": "user", "content": user_msg}
        ]

    def _call_llm(self, messages: List[Dict], temperature: float = 0.4) -> str:
        """调用大模型"""
        # 当客户端是占位实现或调用异常时，返回兜底回答
        if getattr(self.client, 'is_dummy', False):
            return "抱歉，暂时无法连接大模型服务。我已记录您的问题，将尽快回复。"
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "qwen-max"),
                messages=messages,
                temperature=temperature,
                max_tokens=500,
                top_p=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"调用LLM失败，使用兜底回复：{e}")
            return "抱歉，当前服务繁忙，请稍后再试。"


class PriceAgent(BaseAgent):
    """议价处理Agent"""

    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int=0) -> str:
        """重写生成逻辑"""
        dynamic_temp = self._calc_temperature(bargain_count)
        messages = self._build_messages(user_msg, item_desc, context)
        messages[0]['content'] += f"\n▲当前议价轮次：{bargain_count}"
        reply = self._call_llm(messages, temperature=dynamic_temp)
        return self.safety_filter(reply)

    def _calc_temperature(self, bargain_count: int) -> float:
        """动态温度策略"""
        return min(0.3 + bargain_count * 0.15, 0.9)


class TechAgent(BaseAgent):
    """技术咨询Agent"""
    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int=0) -> str:
        """重写生成逻辑"""
        messages = self._build_messages(user_msg, item_desc, context)
        # messages[0]['content'] += "\n▲知识库：\n" + self._fetch_tech_specs()
        reply = self._call_llm(messages, temperature=0.4)
        return self.safety_filter(reply)


    # def _fetch_tech_specs(self) -> str:
    #     """模拟获取技术参数（可连接数据库）"""
    #     return "功率：200W@8Ω\n接口：XLR+RCA\n频响：20Hz-20kHz"


class ClassifyAgent(BaseAgent):
    """意图识别Agent"""

    def generate(self, **args) -> str:
        response = super().generate(**args)
        return response


class DefaultAgent(BaseAgent):
    """默认处理Agent"""

    def _call_llm(self, messages: List[Dict], *args) -> str:
        """限制默认回复长度"""
        response = super()._call_llm(messages, temperature=0.7)
        return response