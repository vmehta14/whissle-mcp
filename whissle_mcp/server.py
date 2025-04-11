"""
Whissle MCP Server

⚠️ IMPORTANT: This server provides access to Whissle API endpoints which may incur costs.
Each tool that makes an API call is marked with a cost warning. Please follow these guidelines:

1. Only use tools when explicitly requested by the user
2. For tools that process audio, consider the length of the audio as it affects costs
3. Some operations like translation or summarization may have higher costs

Tools without cost warnings in their description are free to use as they only read existing data.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from whissle import WhissleClient
from whissle_mcp.utils import (
    make_error,
    make_output_path,
    make_output_file,
    handle_input_file,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("whissle_mcp")

load_dotenv()
auth_token = os.getenv("WHISSLE_AUTH_TOKEN")
base_path = os.getenv("WHISSLE_MCP_BASE_PATH")

if not auth_token:
    raise ValueError("WHISSLE_AUTH_TOKEN environment variable is required")

try:
    client = WhissleClient(auth_token=auth_token).sync_client
    logger.info("Whissle client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Whissle client: {str(e)}")
    raise ValueError(f"Failed to initialize Whissle client: {str(e)}")

mcp = FastMCP("Whissle")


def handle_api_error(error_msg, operation_name, retry_count=0, max_retries=2):
    """Helper function to handle API errors with retries and better error messages"""
    logger.error(f"API error during {operation_name}: {error_msg}")
    
    if "HTTP 500" in error_msg:
        if retry_count < max_retries:
            # Exponential backoff: 2, 4, 8 seconds
            wait_time = 2 ** (retry_count + 1)
            logger.info(f"HTTP 500 error during {operation_name}. Retrying in {wait_time} seconds... (Attempt {retry_count+1}/{max_retries+1})")
            time.sleep(wait_time)
            return None  # Signal to retry
        else:
            # Provide more detailed error message for upload issues
            if "uploading file" in error_msg.lower():
                return make_error(
                    f"Server error during {operation_name}. The file upload to the Whissle API failed. "
                    f"This could be due to:\n"
                    f"1. Temporary server issues\n"
                    f"2. File format compatibility issues\n"
                    f"3. Network connectivity problems\n"
                    f"Please try again later or contact Whissle support. Error: {error_msg}"
                )
            else:
                return make_error(
                    f"Server error during {operation_name}. This might be a temporary issue with the Whissle API. "
                    f"Please try again later or contact Whissle support. Error: {error_msg}"
                )
    elif "HTTP 413" in error_msg:
        return make_error(f"File too large. Please try a smaller file. Error: {error_msg}")
    elif "HTTP 415" in error_msg:
        return make_error(f"Unsupported file format. Please use a supported format. Error: {error_msg}")
    elif "HTTP 401" in error_msg or "HTTP 403" in error_msg:
        return make_error(f"Authentication error. Please check your API token. Error: {error_msg}")
    else:
        return make_error(f"API error during {operation_name}: {error_msg}")


@mcp.tool(
    description="""Convert speech to text with a given model and save the output text file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.

    ⚠️ COST WARNING: This tool makes an API call to Whissle which may incur costs. Only use when explicitly requested by the user.

    Args:
        audio_file_path (str): Path to the audio file to transcribe
        model_name (str, optional): The name of the ASR model to use. Defaults to "en-NER"
        timestamps (bool, optional): Whether to include word timestamps
        boosted_lm_words (List[str], optional): Words to boost in recognition
        boosted_lm_score (int, optional): Score for boosted words (0-100)
        output_directory (str, optional): Directory where files should be saved.
            Defaults to $HOME/Desktop if not provided.

    Returns:
        TextContent with the transcription and path to the output file.
    """
)
def speech_to_text(audio_file_path: str, model_name: str = "en-NER", timestamps: bool = True, boosted_lm_words: List[str] = None, boosted_lm_score: int = 80) -> Dict:
    """Convert speech to text using Whissle API"""
    try:
        # Check if file exists
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return {"error": f"Audio file not found: {audio_file_path}"}
        
        # Check file size
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {audio_file_path}")
            return {"error": f"Audio file is empty: {audio_file_path}"}
        
        # Check file format
        file_ext = os.path.splitext(audio_file_path)[1].lower()
        if file_ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
            logger.error(f"Unsupported audio format: {file_ext}")
            return {"error": f"Unsupported audio format: {file_ext}. Supported formats: wav, mp3, ogg, flac, m4a"}
        
        # Check file size limits
        max_size_mb = 25
        if file_size > max_size_mb * 1024 * 1024:
            logger.error(f"File too large: {file_size / (1024*1024):.2f} MB")
            return {"error": f"File too large ({file_size / (1024*1024):.2f} MB). Maximum size is {max_size_mb} MB."}
        
        # Log the request details
        logger.info(f"Transcribing audio file: {audio_file_path}")
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
        logger.info(f"File format: {file_ext}")
        
        # Try with a different model if the default one fails
        models_to_try = ["en-NER"]
        last_error = None
        
        for try_model in models_to_try:
            retry_count = 0
            max_retries = 2
            
            while retry_count <= max_retries:
                try:
                    logger.info(f"Attempting transcription with model: {try_model} (Attempt {retry_count+1}/{max_retries+1})")
                    response = client.speech_to_text(
                        audio_file_path=audio_file_path,
                        model_name=try_model,
                        timestamps=timestamps,
                        boosted_lm_words=boosted_lm_words,
                        boosted_lm_score=boosted_lm_score
                    )
                    
                    if response and hasattr(response, 'transcript'):
                        logger.info(f"Transcription successful with model: {try_model}")
                        
                        result = {
                            "transcript": response.transcript,
                            "duration_seconds": getattr(response, 'duration_seconds', 0),
                            "language_code": getattr(response, 'language_code', 'en')
                        }
                        
                        if hasattr(response, 'timestamps'):
                            result["timestamps"] = response.timestamps
                        
                        if hasattr(response, 'diarize_output') and response.diarize_output:
                            result["diarize_output"] = response.diarize_output
                        
                        return result
                    else:
                        last_error = "No transcription was returned from the API"
                        logger.error(f"No transcription returned from API with model {try_model}")
                        break
                except Exception as api_error:
                    error_msg = str(api_error)
                    logger.error(f"Error with model {try_model}: {error_msg}")
                    last_error = error_msg
                    
                    error_result = handle_api_error(error_msg, "transcription", retry_count, max_retries)
                    if error_result is not None:
                        if retry_count == max_retries:
                            break
                        else:
                            return {"error": error_result}
                    
                    retry_count += 1
        
        if "HTTP 500" in last_error:
            logger.error(f"All transcription attempts failed with HTTP 500: {last_error}")
            return {"error": f"Server error during transcription. This might be a temporary issue with the Whissle API. Please try again later or contact Whissle support. Error: {last_error}"}
        else:
            logger.error(f"All transcription attempts failed: {last_error}")
            return {"error": f"Failed to transcribe audio: {last_error}"}
            
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {str(e)}")
        return {"error": f"Failed to transcribe audio: {str(e)}"}


@mcp.tool(
    description="""Convert speech to text with speaker diarization and save the output text file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.

    ⚠️ COST WARNING: This tool makes an API call to Whissle which may incur costs. Only use when explicitly requested by the user.

    Args:
        audio_file_path (str): Path to the audio file to transcribe
        model_name (str, optional): The name of the ASR model to use. Defaults to "en-NER"
        max_speakers (int, optional): Maximum number of speakers to identify
        boosted_lm_words (List[str], optional): Words to boost in recognition
        boosted_lm_score (int, optional): Score for boosted words (0-100)
        output_directory (str, optional): Directory where files should be saved.
            Defaults to $HOME/Desktop if not provided.

    Returns:
        TextContent with the diarized transcription and path to the output file.
    """
)
def diarize_speech(audio_file_path: str, model_name: str = "en-NER", max_speakers: int = 2, boosted_lm_words: List[str] = None, boosted_lm_score: int = 80) -> Dict:
    """Diarize speech using Whissle API"""
    try:
        # Check if file exists
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return {"error": f"Audio file not found: {audio_file_path}"}
        
        # Check file size
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {audio_file_path}")
            return {"error": f"Audio file is empty: {audio_file_path}"}
        
        # Check file format
        file_ext = os.path.splitext(audio_file_path)[1].lower()
        if file_ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
            logger.error(f"Unsupported audio format: {file_ext}")
            return {"error": f"Unsupported audio format: {file_ext}. Supported formats: wav, mp3, ogg, flac, m4a"}
        
        # Check file size limits
        max_size_mb = 25
        if file_size > max_size_mb * 1024 * 1024:
            logger.error(f"File too large: {file_size / (1024*1024):.2f} MB")
            return {"error": f"File too large ({file_size / (1024*1024):.2f} MB). Maximum size is {max_size_mb} MB."}
        
        # Log the request details
        logger.info(f"Diarizing audio file: {audio_file_path}")
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
        logger.info(f"File format: {file_ext}")
        
        # Try with a different model if the default one fails
        models_to_try = ["en-NER"]
        last_error = None
        
        for try_model in models_to_try:
            retry_count = 0
            max_retries = 2
            
            while retry_count <= max_retries:
                try:
                    logger.info(f"Attempting diarization with model: {try_model} (Attempt {retry_count+1}/{max_retries+1})")
                    response = client.diarize_stt(
                        audio_file_path=audio_file_path,
                        model_name=try_model,
                        max_speakers=max_speakers,
                        boosted_lm_words=boosted_lm_words,
                        boosted_lm_score=boosted_lm_score
                    )
                    
                    if response and hasattr(response, 'diarize_output') and response.diarize_output:
                        logger.info(f"Diarization successful with model: {try_model}")
                        
                        result = {
                            "transcript": getattr(response, 'transcript', ''),
                            "duration_seconds": getattr(response, 'duration_seconds', 0),
                            "language_code": getattr(response, 'language_code', 'en'),
                            "diarize_output": response.diarize_output
                        }
                        
                        if hasattr(response, 'timestamps'):
                            result["timestamps"] = response.timestamps
                        
                        return result
                    else:
                        last_error = "No diarized transcription was returned from the API"
                        logger.error(f"No diarized transcription returned from API with model {try_model}")
                        break
                except Exception as api_error:
                    error_msg = str(api_error)
                    logger.error(f"Error with model {try_model}: {error_msg}")
                    last_error = error_msg
                    
                    error_result = handle_api_error(error_msg, "diarization", retry_count, max_retries)
                    if error_result is not None:
                        if retry_count == max_retries:
                            break
                        else:
                            return {"error": error_result}
                    
                    retry_count += 1
        
        if "HTTP 500" in last_error:
            logger.error(f"All diarization attempts failed with HTTP 500: {last_error}")
            return {"error": f"Server error during diarization. This might be a temporary issue with the Whissle API. Please try again later or contact Whissle support. Error: {last_error}"}
        else:
            logger.error(f"All diarization attempts failed: {last_error}")
            return {"error": f"Failed to diarize speech: {last_error}"}
            
    except Exception as e:
        logger.error(f"Unexpected error during diarization: {str(e)}")
        return {"error": f"Failed to diarize speech: {str(e)}"}


@mcp.tool(
    description="""Translate text from one language to another.

    ⚠️ COST WARNING: This tool makes an API call to Whissle which may incur costs. Only use when explicitly requested by the user.

    Args:
        text (str): The text to translate
        source_language (str): Source language code (e.g., "en" for English)
        target_language (str): Target language code (e.g., "es" for Spanish)

    Returns:
        TextContent with the translated text.
    """
)
def translate_text(
    text: str,
    source_language: str,
    target_language: str,
) -> TextContent:
    try:
        if not text:
            logger.error("Empty text provided for translation")
            return make_error("Text is required")
        
        # Log the request details
        logger.info(f"Translating text from {source_language} to {target_language}")
        logger.info(f"Text length: {len(text)} characters")
        
        retry_count = 0
        max_retries = 2  # Increased from 1 to 2
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Attempting translation (Attempt {retry_count+1}/{max_retries+1})")
                response = client.machine_translation(
                    text=text,
                    source_language=source_language,
                    target_language=target_language,
                )
                
                if response and response.translated_text:
                    logger.info("Translation successful")
                    return TextContent(
                        type="text",
                        text=f"Translation:\n{response.translated_text}",
                    )
                else:
                    logger.error("No translation was returned from the API")
                    return make_error("No translation was returned from the API")
            except Exception as api_error:
                error_msg = str(api_error)
                logger.error(f"Translation error: {error_msg}")
                
                # Handle API errors with retries
                error_result = handle_api_error(error_msg, "translation", retry_count, max_retries)
                if error_result is not None:  # If we should not retry
                    return error_result  # Return the error message
                
                retry_count += 1
        
        # If we get here, all retries failed
        logger.error(f"All translation attempts failed after {max_retries+1} attempts")
        return make_error(f"Failed to translate text after {max_retries+1} attempts")
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        return make_error(f"Failed to translate text: {str(e)}")


@mcp.tool(
    description="""Summarize text using an LLM model.

    ⚠️ COST WARNING: This tool makes an API call to Whissle which may incur costs. Only use when explicitly requested by the user.

    Args:
        content (str): The text to summarize
        model_name (str, optional): The LLM model to use. Defaults to "openai"
        instruction (str, optional): Specific instructions for summarization

    Returns:
        TextContent with the summary.
    """
)
def summarize_text(
    content: str,
    model_name: str = "openai",
    instruction: Optional[str] = None,
) -> TextContent:
    try:
        if not content:
            logger.error("Empty content provided for summarization")
            return make_error("Content is required")
        
        # Log the request details
        logger.info(f"Summarizing text using model: {model_name}")
        logger.info(f"Text length: {len(content)} characters")
        
        retry_count = 0
        max_retries = 2  # Increased from 1 to 2
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Attempting summarization (Attempt {retry_count+1}/{max_retries+1})")
                response = client.llm_text_summarizer(
                    content=content,
                    model_name=model_name,
                    instruction=instruction,
                )
                
                if response and response.response:
                    logger.info("Summarization successful")
                    return TextContent(
                        type="text",
                        text=f"Summary:\n{response.response}",
                    )
                else:
                    logger.error("No summary was returned from the API")
                    return make_error("No summary was returned from the API")
            except Exception as api_error:
                error_msg = str(api_error)
                logger.error(f"Summarization error: {error_msg}")
                
                # Handle API errors with retries
                error_result = handle_api_error(error_msg, "summarization", retry_count, max_retries)
                if error_result is not None:  # If we should not retry
                    return error_result  # Return the error message
                
                retry_count += 1
        
        # If we get here, all retries failed
        logger.error(f"All summarization attempts failed after {max_retries+1} attempts")
        return make_error(f"Failed to summarize text after {max_retries+1} attempts")
    except Exception as e:
        logger.error(f"Unexpected error during summarization: {str(e)}")
        return make_error(f"Failed to summarize text: {str(e)}")


@mcp.tool(
    description="List all available ASR models and their capabilities."
)
def list_asr_models() -> TextContent:
    """List all available ASR models.

    Returns:
        TextContent with a formatted list of available models
    """
    try:
        logger.info("Fetching available ASR models...")
        
        retry_count = 0
        max_retries = 2  # Increased from 1 to 2
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Attempting to list models (Attempt {retry_count+1}/{max_retries+1})")
                models = client.list_asr_models()
                
                if not models:
                    logger.error("No models were returned from the API")
                    return make_error("No models were returned from the API")

                # Handle both string and object responses
                if isinstance(models, list):
                    if all(isinstance(model, str) for model in models):
                        # If models is a list of strings
                        model_list = "\n".join(f"Model: {model}" for model in models)
                    else:
                        # If models is a list of objects with name and description
                        model_list = "\n".join(
                            f"Model: {model.name}\nDescription: {model.description}\n"
                            for model in models
                        )
                else:
                    logger.error("Unexpected response format from API")
                    return make_error("Unexpected response format from API")

                logger.info("Successfully retrieved ASR models")
                return TextContent(
                    type="text",
                    text=f"Available ASR Models:\n\n{model_list}",
                )
            except Exception as api_error:
                error_msg = str(api_error)
                logger.error(f"Error listing models: {error_msg}")
                
                # Handle API errors with retries
                error_result = handle_api_error(error_msg, "listing models", retry_count, max_retries)
                if error_result is not None:  # If we should not retry
                    return error_result  # Return the error message
                
                retry_count += 1
        
        # If we get here, all retries failed
        logger.error(f"All attempts to list models failed after {max_retries+1} attempts")
        return make_error(f"Failed to list ASR models after {max_retries+1} attempts")
    except Exception as e:
        logger.error(f"Unexpected error listing ASR models: {str(e)}")
        return make_error(f"Failed to list ASR models: {str(e)}")


def main():
    print("Starting Whissle MCP server")
    """Run the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main() 