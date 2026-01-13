"""Google Cloud Text-to-Speech provider implementation for single speaker."""

from google.cloud import texttospeech
from typing import List
from ..base import TTSProvider
import logging

logger = logging.getLogger(__name__)

class GeminiTTS(TTSProvider):
    """Google Cloud Text-to-Speech provider for single speaker."""
    
    def __init__(self, api_key: str = None, model: str = "en-US-Journey-F"):
        """
        Initialize Google Cloud TTS provider.
        
        Args:
            api_key (str): Google Cloud API key
            model (str): Default voice model to use
        """
        self.model = model
        try:
            self.client = texttospeech.TextToSpeechClient()
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS client: {str(e)}")
            raise

    def generate_audio(self, text: str, voice: str = "en-US-Journey-F", 
                      model: str = None, **kwargs) -> bytes:
        """
        Generate audio using Google Cloud TTS API.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice ID/name to use (format: "{language-code}-{name}-{gender}")
            model (str): Optional model override
            
        Returns:
            bytes: Audio data
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If audio generation fails
        """
        self.validate_parameters(text, voice, model or self.model)
        
        try:
            # Create synthesis input
            # TODO prompt
            prompt = """
            你是一位中文 Podcast《AI 工程實驗室》的專業主持人，聲線自然、有精神但不浮誇，語氣友善、帶一點工程師式的理性幽默。
請用清晰咬字與乾淨收尾，整體節奏中等偏快；但在一個完整的段落時，應該適時的有所停緩，必免有念搞的感覺。
遇到重點句（例如痛點、優勢、最佳實踐、結尾 call-to-action）要更有力、稍微加重語氣。
英文產品名/工具名（markitdown、Markdown、LLM、Excel、PowerPoint、Word、PDF、pandoc、unstructured.io）請念得清楚，專有名詞保持一致。
            """
            synthesis_input = texttospeech.SynthesisInput(
                prompt=prompt,
                text=text
            )
            
            # Parse language code from voice ID (e.g., "en-IN" from "en-IN-Journey-D")
            #language_code = "-".join(voice.split("-")[:2])
            language_code = "cmn-tw"
            #language_code = "en-US"
            voice = 'Charon'

            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice,
                model_name ="gemini-2.5-flash-preview-tts"
            #model_name = "gemini-2.5-flash-tts"
            )
            
            # Set audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            # Generate speech
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )
            
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Failed to generate audio: {str(e)}")
            raise RuntimeError(f"Failed to generate audio: {str(e)}") from e
    
    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        return self.COMMON_SSML_TAGS
        
    def validate_parameters(self, text: str, voice: str, model: str) -> None:
        """
        Validate input parameters before generating audio.
        
        Args:
            text (str): Input text
            voice (str): Voice ID/name
            model (str): Model name
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().validate_parameters(text, voice, model)
        
        if not text:
            raise ValueError("Text cannot be empty")
        
        if not voice:
            raise ValueError("Voice must be specified")