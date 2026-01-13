"""Google Cloud Text-to-Speech provider implementation."""
import re
import time
import random

from typing import List
from ..base import TTSProvider
from google import genai
from google.genai import types

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class GeminiMultiTTS2(TTSProvider):
    """Google Cloud Text-to-Speech provider with multi-speaker support."""

    def __init__(self, api_key: str, model: str):
        # logger.info(f"GeminiMultiTTS2: key - {api_key}, model - {model}")

        self.model = model
        # api_key = os.environ.get("GEMINI_API_KEY")
        # logger.info(f"GeminiMultiTTS2: key - {api_key}")
        self.client = genai.Client(api_key=api_key)

        """
        #Initialize Google Cloud TTS provider.
        #
        #Args:
        #    api_key (str): Google Cloud API key
        #"""
        # self.model = model
        # try:
        #    self.client = texttospeech.TextToSpeechClient(
        #        client_options={'api_key': api_key} if api_key else None
        #    )
        #    logger.info("Successfully initialized GeminiMultiTTS client")
        # except Exception as e:
        #    logger.error(f"Failed to initialize GeminiMultiTTS client: {str(e)}")
        #    raise

    from google import genai
    from google.genai import types

    def generate_tts_prompt(self, text: str, gender_order: str) -> str:
        """
        將 Podcast 腳本轉換為 Gemini TTS 格式的 prompt

        Args:
            text: 原始腳本，格式為 "x: ...\ny: ...\nx: ..."
            gender_order: 性別順序描述，例如 "x是男生，y是女生" 或 "x是女生，y是男生"
        Returns:
            str: 優化後的 TTS prompt
        """
        start_time = time.time()
        logger.info("Starting generate_tts_prompt")

        system_prompt = f"""
        你的任務是將初版 Podcast 對話腳本轉換為符合 Gemini TTS API 標準格式的完整 prompt。

    ### 輸入格式
    你會收到以下格式的 Podcast 對話腳本：
    ```
    x: 對話內容
    y: 對話內容
    x: 對話內容
    ```

    其中 x 和 y 代表節目中的兩位主持人/來賓。
    注意性別！ {gender_order} ，取名時務必根據性別取適合的名字
    ### 輸出格式要求
    你必須按照以下**精確結構**生成完整的 TTS prompt：
    ```
    # THE SCENE: 
    [Podcast 錄製場景描述：包含錄音環境、節目與對話氛圍、主持人狀態、情緒，以確立基調和氛圍]
    
    # AUDIO PROFILE: x
    ## PODCAST ROLE & NAME
    [x 在節目中的角色與名稱，ex: 主持人/產業分析師，以及 x 被稱呼的名字]
    
    ## DIRECTOR'S NOTES
    [包含具體的成效指引，不要過度指定，只定義對效能有重要影響的項目，避免模型創意受限，包括以下]
    Style: [詳細描述 x 的說話風格、語氣、情感表達方式]
    Pacing: [描述 x 的語速和節奏變化]
    Accent: [具體的口音描述]
    
    # AUDIO PROFILE: y
    ## PODCAST ROLE & NAME
    ## DIRECTOR'S NOTES
    Style: 
    Pacing: 
    Accent: 

    # TRANSCRIPT
    TTS the following conversation between x and y:
x: [對話內容]
y: [對話內容]
x: [對話內容]
y: [對話內容]
    ```
### 具體執行步驟

**步驟 1：分析 Podcast 腳本並命名**
- 閱讀完整的 x 和 y 對話
- 判斷節目類型（教育型、訪談型、新聞評論等）
- 識別 x 和 y 在節目中的角色定位、符合專業度，並給予適合風格的名字。注意 x, y 的性別給適合的名字，不要太土，中文節目也可英文名


**步驟 2：處理對話中的稱呼替換**
在 TRANSCRIPT 部分，**必須**處理以下替換：
**自我介紹替換：**
- 原文對話中如出現名稱，請替代為你取的名字
- 範例：「大家好我是主持人 XXX」(XXX 是您取的名字)

**互相稱呼替換：**
- 如果對話中出現稱呼對方角色的情況，替換為你為對方取的名字
- 範例：「分析師你覺得呢？」→ 「XXX 你覺得呢？」(XXX 是您取的名字)
- 範例：「主持人剛才提到」→ 「XXX 剛才提到」(XXX 是您取的名字)

**保持自然：**
- 如果原文沒有自稱或互相稱呼，不需要強制加入名字
- 替換後的對話必須聽起來自然流暢

**步驟 3：建構 Podcast 場景**

在 `## THE SCENE:` 部分描述：
- Podcast 節目的定位和風格
- 錄音環境（專業錄音室、居家錄音等）
- 節目的整體氛圍和調性
- 主持人/來賓之間的互動動態

**步驟 4：定義 Audio Profile**

為 **x** 和 **y** 各自定義：

**PODCAST ROLE & NAME**：節目中的身份與 被稱呼的名字
**DIRECTOR'S NOTES**：包含 Style, Pacing, Accent 的完整描述

### 重要原則

1. **格式嚴格遵守**：TRANSCRIPT 部分只能使用 x: 和 y: 標記
2. **Podcast 場景一致**：所有描述必須符合 Podcast 錄製情境
3. **具體優於籠統**：風格描述要具體，避免單一形容詞
4. **自然留白**：不要過度指定細節，讓模型有創意空間
5. **名字替換完整**：確保對話中的自稱和互相稱呼都已正確替換

### 輸出要求

直接輸出完整的 TTS prompt，不要有任何前言或後綴說明。
從 `# THE SCENE: ` 開始，到 TRANSCRIPT 的最後一行對話結束。
    """
        # model = 'gemini-3-flash-preview'
        model = 'gemini-2.5-flash-lite-preview-09-2025'
        response = self.client.models.generate_content(
            model=model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,  # 可調整創意度
            )
        )
        tts_prompt = response.text

        elapsed_time = time.time() - start_time
        logger.info("generate_tts_prompt completed in %.2f seconds", elapsed_time)

        return tts_prompt

    def convert_xml_to_simple_format(self, script: str) -> str:
        """
        將 XML 標籤格式的對話腳本轉換為 x:/y: 格式

        Args:
            script: XML 格式的腳本，如 <Person1>"對話"</Person1><Person2>"對話"</Person2>

        Returns:
            str: x:/y: 格式的腳本
        """
        start_time = time.time()
        logger.info("Starting convert_xml_to_simple_format")

        # 移除多餘的空白和換行，將所有內容合併為一行處理
        script = script.strip()

        # 使用正則表達式匹配 <Person1>...</Person1> 或 <Person2>...</Person2>
        # 包括跨行的情況
        pattern = r'<(Person\d+)>\s*"?(.*?)"?\s*</\1>'

        matches = re.findall(pattern, script, re.DOTALL)

        # 建立 Person 到 x/y 的映射
        person_map = {}
        labels = ['x', 'y']
        label_idx = 0

        result_lines = []

        for person_tag, content in matches:
            # 清理內容：移除前後空白、移除多餘的引號
            content = content.strip().strip('"').strip("'")

            # 如果這個 Person 還沒有映射，分配一個新的標籤
            if person_tag not in person_map:
                if label_idx < len(labels):
                    person_map[person_tag] = labels[label_idx]
                    label_idx += 1
                else:
                    # 如果超過兩個說話者，繼續用 x, y 循環
                    person_map[person_tag] = labels[label_idx % len(labels)]
                    label_idx += 1

            # 生成對應的行
            label = person_map[person_tag]
            result_lines.append(f"{label}: {content}")

        result = '\n'.join(result_lines)

        elapsed_time = time.time() - start_time
        logger.info("convert_xml_to_simple_format completed in %.2f seconds", elapsed_time)

        return result

    def generate_audio(self, text: str, voice: str, model: str,
                       voice2: str, ending_message: str = ""):
        start_time = time.time()
        logger.info("Starting generate_audio")
        logger.info("voice:%s , voice2:%s- model:%s, ending:%s", voice, voice2, model, ending_message)

        text = self.convert_xml_to_simple_format(text)
        logger.info("Text:\n%s", text)
        man_voices = ['Orus', 'Aoede', 'Autonoe', 'Iapetus', 'Erinome', 'Rasalgethi', 'Algenib', 'Schedar', 'Gacrux',
                      'Achird', 'Zubenelgenubi', 'Sadachbia', 'Sadaltager']
        woman_voices = ['Charon', 'Callirrhoe', 'Despina', 'Achernar', 'Laomedeia', 'Vindemiatrix', 'Sulafat']

        # 隨機選出一男一女的聲音
        selected_man = random.choice(man_voices)
        selected_woman = random.choice(woman_voices)

        # 隨機決定男女順序
        is_man_first = random.choice([True, False])

        if is_man_first:
            voice = selected_man
            voice2 = selected_woman
            gender_order = "x 是男生，y 是女生"
        else:
            voice = selected_woman
            voice2 = selected_man
            gender_order = "x 是女生，y 是男生"

        logger.info("Selected voices: voice=%s, voice2=%s, order=%s", voice, voice2, gender_order)

        tts_prompt = self.generate_tts_prompt(text, gender_order)

        t2 = time.time()
        elapsed_time = t2 - start_time
        logger.info("tts prompt completed in %.2f seconds", elapsed_time)
        logger.info("TTS prompt:\n%s", tts_prompt)
        # prompt = """
        #         TTS the following conversation between x and y:
        #         x: 歡迎收聽市場解碼 - 深入淺出，解讀全球產業動態。我是產業分析師，今天要跟大家聊一個很震撼的話題。
        # y: 嗨！我是投資人，最近看新聞說酒類產業好像出了大問題？
        # x: 沒錯，而且問題相當嚴重。想像一下，8300億美元，你知道這是多少錢嗎？
        # y: 天哪...那是...多少？
        # x: 差不多是一個中型國家的GDP！過去四年，全球酒類產業的市值就這樣蒸發了。主要指數從2021年的高點，跌了46%。
        # y: 等等，46%？那真的是...很多人的投資組合應該被打得很慘。
        # x: 完全同意。但這不是一夜之間發生的，而是多重壓力疊加的結果。讓我給你舉個例子，就像你騎機車，突然同時遇到紅燈、下雨、還有塞車，對吧？
        # y: 哈，那確實會很慘。那酒類產業遇到了什麼『紅燈』？
        # x: 首先，健康意識抬頭。美國衛生署發出警告，世界衛生組織也在說飲酒的風險。然後，社會風氣改變了，年輕人開始覺得飲酒不時尚。
        # y: 不時尚？真的嗎？我以為喝酒一直都很...社交啊。
        #         """
        # voice = 'Zephyr'
        # voice2 = 'Puck'
        # https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-tts
        # December 2025
        # model = "gemini-2.5-pro-preview-tts"
        # December 2025
        # https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-tts
        # model = "gemini-2.5-flash-preview-tts"
        response = self.client.models.generate_content(
            model=model,
            contents=tts_prompt,
            config=types.GenerateContentConfig(
                temperature=1,
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker='x',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=voice,
                                    )
                                )
                            ),
                            types.SpeakerVoiceConfig(
                                speaker='y',
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=voice2,
                                    )
                                )
                            ),
                        ]
                    )
                )
            )
        )
        blob = response.candidates[0].content.parts[0].inline_data
        logger.info("Mime:%s", blob.mime_type)
        t3 = time.time()
        elapsed_time = t3 - t2
        logger.info("Audio generated completed in %.2f seconds", elapsed_time)
        return blob.data

    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        # Add any Google-specific SSML tags to the common ones
        return self.COMMON_SSML_TAGS

    def validate_parameters(self, text: str, voice: str, model: str) -> None:
        """
        Validate input parameters before generating audio.
        
        Args:
            text (str): Input text
            voice (str): Voice ID
            model (str): Model name
            
        Raises:
            ValueError: If parameters are invalid
        """
        logger.debug(f"Validating parameters: {text}, {voice}, {model}")
        super().validate_parameters(text, voice, model)

        # Additional validation for multi-speaker model
        if model != "en-US-Studio-MultiSpeaker":
            raise ValueError(
                "Google Multi-speaker TTS requires model='en-US-Studio-MultiSpeaker'"
            )
