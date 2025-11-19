import unittest
import pytest
import tempfile
import os
from podcastfy.content_generator import ContentGenerator
from podcastfy.utils.config import Config


MOCK_IMAGE_PATHS = [
    "https://raw.githubusercontent.com/souzatharsis/podcastfy/refs/heads/main/data/images/Senecio.jpeg",
    "https://raw.githubusercontent.com/souzatharsis/podcastfy/refs/heads/main/data/images/connection.jpg",
]

MODEL_NAME = "gemini-2.5-flash"
API_KEY_LABEL = "GEMINI_API_KEY"


# TODO: Should be a fixture
def sample_conversation_config():
    conversation_config = {
        "word_count": 500,
        "roles_person1": "professor",
        "roles_person2": "student",
        "podcast_name": "Teachfy",
        "podcast_tagline": "Learning Through Conversation",
    }
    return conversation_config


class TestGenAIPodcast(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        config = Config()
        self.api_key = config.GEMINI_API_KEY
        self.config = config

    def test_generate_qa_content(self):
        """
        Test the generate_qa_content method of ContentGenerator.
        """
        content_generator = ContentGenerator(model_name=MODEL_NAME, api_key_label=API_KEY_LABEL)
        input_text = "United States of America"
        result = content_generator.generate_qa_content(input_text)
        self.assertIsNotNone(result)
        self.assertNotEqual(result, "")
        self.assertIsInstance(result, str)

    def test_custom_conversation_config(self):
        """
        Test the generation of content using a custom conversation configuration file.
        """
        conversation_config = sample_conversation_config()
        content_generator = ContentGenerator(model_name=MODEL_NAME, api_key_label=API_KEY_LABEL, conversation_config=conversation_config)
        input_text = "United States of America"

        result = content_generator.generate_qa_content(input_text)

        self.assertIsNotNone(result)
        self.assertNotEqual(result, "")
        self.assertIsInstance(result, str)

        # Check for elements from the custom config
        self.assertIn(conversation_config["podcast_name"].lower(), result.lower())
        self.assertIn(conversation_config["podcast_tagline"].lower(), result.lower())

    def test_generate_qa_content_from_images(self):
        """Test generating Q&A content from two input images."""
        image_paths = MOCK_IMAGE_PATHS

        content_generator = ContentGenerator(model_name=MODEL_NAME, api_key_label=API_KEY_LABEL)

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        ) as temp_file:
            result = content_generator.generate_qa_content(
                input_texts="",  # Empty string for input_texts
                image_file_paths=image_paths,
                output_filepath=temp_file.name,
            )

        self.assertIsNotNone(result)
        self.assertNotEqual(result, "")
        self.assertIsInstance(result, str)

        # Check if the output file was created and contains the same content
        with open(temp_file.name, "r") as f:
            file_content = f.read()

        self.assertEqual(result, file_content)

        # Clean up the temporary file
        os.unlink(temp_file.name)


    def test_generate_qa_content_from_raw_text(self):
        """Test generating Q&A content from raw input text."""
        raw_text = "The wonderful world of LLMs."
        content_generator = ContentGenerator(model_name=MODEL_NAME, api_key_label=API_KEY_LABEL)

        result = content_generator.generate_qa_content(input_texts=raw_text)

        self.assertIsNotNone(result)
        self.assertNotEqual(result, "")
        self.assertIsInstance(result, str)



if __name__ == "__main__":
    unittest.main()
