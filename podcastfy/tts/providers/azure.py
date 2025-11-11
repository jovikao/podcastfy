"""Azure Cognitive Services Speech TTS provider implementation."""

import os
import azure.cognitiveservices.speech as speechsdk
from typing import List, Optional
from ..base import TTSProvider


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
            Audio data as bytes in WAV format

        Raises:
            RuntimeError: If audio generation fails
        """
        self.validate_parameters(text, voice, model)

        try:
            # Set the voice for this specific generation
            self.speech_config.speech_synthesis_voice_name = voice

            # Create synthesizer with null output (we'll get audio from result)
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None  # None means no audio output, we get data from result
            )

            # Synthesize text to speech
            result = speech_synthesizer.speak_text_async(text).get()

            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Return audio data as bytes
                return bytes(result.audio_data)
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                error_msg = f"Speech synthesis canceled: {cancellation_details.reason}"
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    error_msg += f"\nError details: {cancellation_details.error_details}"
                raise RuntimeError(error_msg)
            else:
                raise RuntimeError(f"Unexpected result reason: {result.reason}")

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Failed to generate audio with Azure TTS: {str(e)}") from e
