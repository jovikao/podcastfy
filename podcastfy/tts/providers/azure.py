"""Azure Cognitive Services Speech TTS provider implementation."""

import os
import logging
import tempfile
import azure.cognitiveservices.speech as speechsdk
from typing import List, Optional
from ..base import TTSProvider

logger = logging.getLogger(__name__)


class AzureTTS(TTSProvider):
    """Azure Cognitive Services Text-to-Speech provider."""

    # Azure supports extensive SSML tags
    PROVIDER_SSML_TAGS: List[str] = [
        'lang', 'p', 'phoneme', 's', 'sub', 'break', 'emphasis',
        'prosody', 'say-as', 'audio', 'bookmark', 'mstts:express-as'
    ]

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize Azure TTS provider.

        Args:
            api_key: Azure Speech subscription key. If None, expects AZURE_API_KEY env variable
            model: Not used for Azure (kept for interface compatibility)

        Environment Variables:
            AZURE_API_KEY: Azure Speech subscription key
            AZURE_SPEECH_REGION: Azure service region (e.g., "eastus", "westus")

        Raises:
            ValueError: If API key or region is not provided
        """
        # Get subscription key
        self.subscription_key = api_key or os.environ.get('AZURE_API_KEY')
        if not self.subscription_key:
            raise ValueError(
                "Azure Speech subscription key must be provided via api_key parameter "
                "or AZURE_SPEECH_KEY environment variable"
            )

        # Get service region
        self.region = os.environ.get('AZURE_SPEECH_REGION', 'eastus')
        if not self.region:
            raise ValueError(
                "Azure Speech region must be set via AZURE_SPEECH_REGION environment variable"
            )

        # Create speech config
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.subscription_key,
            region=self.region
        )

        # Set output format to MP3 (48kHz, 192kbps)
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
        )
        logger.info(f"Azure TTS initialized with region: {self.region}, output format: MP3 48kHz 192kbps")

        # Create synthesizer once for reuse (optimization)
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None  # None means no audio output, we get data from result
        )

        # Store model parameter for interface compatibility (not used in Azure)
        self.model = model

    def get_supported_tags(self) -> List[str]:
        """Get all supported SSML tags including Azure-specific ones."""
        return self.PROVIDER_SSML_TAGS

    def generate_audio(self, text: str, voice: str, model: str, voice2: str = None) -> bytes:
        """
        Generate audio using Azure Cognitive Services Speech API.

        Args:
            text: Text to convert to speech
            voice: Voice name to use (e.g., "en-US-JennyNeural", "en-US-GuyNeural")
            model: Not used for Azure (kept for interface compatibility)
            voice2: Not used for Azure (kept for interface compatibility)

        Returns:
            Audio data as bytes in MP3 format

        Raises:
            RuntimeError: If audio generation fails
        """
        self.validate_parameters(text, voice, model)

        logger.info(f"Generating audio with Azure TTS - Voice: {voice}, Text length: {len(text)} chars")

        try:
            # Set the voice for this specific generation
            self.speech_config.speech_synthesis_voice_name = voice

            # Synthesize text to speech (using reusable synthesizer)
            result = self.speech_synthesizer.speak_text_async(text).get()

            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Get audio data as bytes
                audio_data = bytes(result.audio_data)

                # Log audio data details
                logger.info(f"Audio generated successfully - Size: {len(audio_data)} bytes, Type: {type(audio_data)}")

                # Validate it's MP3 by checking magic bytes (ID3 tag or MPEG sync)
                is_mp3 = self._validate_mp3_format(audio_data)
                logger.info(f"Audio format validation - Is MP3: {is_mp3}")

                if not is_mp3:
                    logger.warning("Audio data does not appear to be MP3 format based on magic bytes")

                # Save to temp file for verification/debugging
                self._save_debug_audio(audio_data, voice)

                return audio_data

            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                error_msg = f"Speech synthesis canceled: {cancellation_details.reason}"
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    error_msg += f"\nError details: {cancellation_details.error_details}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                error_msg = f"Unexpected result reason: {result.reason}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            logger.error(f"Failed to generate audio with Azure TTS: {str(e)}")
            raise RuntimeError(f"Failed to generate audio with Azure TTS: {str(e)}") from e

    def _validate_mp3_format(self, audio_data: bytes) -> bool:
        """
        Validate if audio data is in MP3 format by checking magic bytes.

        Args:
            audio_data: Audio data to validate

        Returns:
            True if data appears to be MP3 format, False otherwise
        """
        if len(audio_data) < 3:
            return False

        # Check for ID3 tag (ID3v2)
        if audio_data[:3] == b'ID3':
            return True

        # Check for MPEG sync bits (0xFF 0xFB or 0xFF 0xFA or 0xFF 0xF3 or 0xFF 0xF2)
        if audio_data[0] == 0xFF and audio_data[1] in [0xFB, 0xFA, 0xF3, 0xF2]:
            return True

        return False

    def _save_debug_audio(self, audio_data: bytes, voice: str) -> None:
        """
        Save audio data to temp file for debugging/verification.

        Args:
            audio_data: Audio data to save
            voice: Voice name (used in filename)
        """
        try:
            # Create temp file with descriptive name
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"azure_tts_debug_{voice.replace(':', '_')}.mp3")

            with open(temp_file, 'wb') as f:
                f.write(audio_data)

            logger.info(f"Debug audio saved to: {temp_file}")
        except Exception as e:
            logger.warning(f"Failed to save debug audio: {str(e)}")
