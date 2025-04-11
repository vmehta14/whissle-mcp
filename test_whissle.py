from whissle import WhissleClient
import os
import time
import logging
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("whissle_test")

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
                logger.error(f"Server error during {operation_name}. The file upload to the Whissle API failed.")
                return f"Server error during {operation_name}. The file upload to the Whissle API failed. This could be due to temporary server issues, file format compatibility issues, or network connectivity problems. Error: {error_msg}"
            else:
                logger.error(f"Server error during {operation_name}. This might be a temporary issue with the Whissle API.")
                return f"Server error during {operation_name}. This might be a temporary issue with the Whissle API. Please try again later or contact Whissle support. Error: {error_msg}"
    elif "HTTP 413" in error_msg:
        logger.error(f"File too large for {operation_name}.")
        return f"File too large. Please try a smaller file. Error: {error_msg}"
    elif "HTTP 415" in error_msg:
        logger.error(f"Unsupported file format for {operation_name}.")
        return f"Unsupported file format. Please use a supported format. Error: {error_msg}"
    elif "HTTP 401" in error_msg or "HTTP 403" in error_msg:
        logger.error(f"Authentication error for {operation_name}.")
        return f"Authentication error. Please check your API token. Error: {error_msg}"
    else:
        logger.error(f"API error during {operation_name}.")
        return f"API error during {operation_name}: {error_msg}"

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for authentication token
    auth_token = os.getenv("WHISSLE_AUTH_TOKEN")
    if not auth_token:
        auth_token = input("Please enter your Whissle authentication token: ")
        if not auth_token:
            logger.error("Authentication token is required to use the Whissle API.")
            print("Error: Authentication token is required to use the Whissle API.")
            return
    
    # Initialize the Whissle client with the auth token
    try:
        logger.info("Initializing Whissle client...")
        client = WhissleClient(auth_token=auth_token).sync_client
        logger.info("Whissle client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Whissle client: {str(e)}")
        print(f"Error: Failed to initialize Whissle client: {str(e)}")
        return
    
    def test_list_models():
        print("\n=== Testing List ASR Models ===")
        logger.info("Testing List ASR Models")
        
        retry_count = 0
        max_retries = 2
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Attempting to list models (Attempt {retry_count+1}/{max_retries+1})")
                models = client.list_asr_models()
                
                if not models:
                    logger.error("No models were returned from the API")
                    print("Error: No models were returned from the API")
                    return
                
                print("Available ASR Models:")
                for model in models:
                    print(f"- {model}")
                
                logger.info("Successfully retrieved ASR models")
                return
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error listing models: {error_msg}")
                
                # Handle API errors with retries
                error_result = handle_api_error(error_msg, "listing models", retry_count, max_retries)
                if error_result is not None:  # If we should not retry
                    print(f"Error: {error_result}")
                    return
                
                retry_count += 1
        
        # If we get here, all retries failed
        logger.error(f"All attempts to list models failed after {max_retries+1} attempts")
        print(f"Error: Failed to list ASR models after {max_retries+1} attempts")
    
    def test_speech_to_text(file_path="test.wav"):
        print("\n=== Testing Speech to Text ===")
        logger.info(f"Testing Speech to Text with file: {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                print(f"Error: Audio file not found: {file_path}")
                return
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"Audio file is empty: {file_path}")
                print(f"Error: Audio file is empty: {file_path}")
                return
            
            # Check file format
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
                logger.error(f"Unsupported audio format: {file_ext}")
                print(f"Error: Unsupported audio format: {file_ext}. Supported formats: wav, mp3, ogg, flac, m4a")
                return
            
            # Check file size limits (if file is too large)
            max_size_mb = 25  # Assuming a reasonable limit
            if file_size > max_size_mb * 1024 * 1024:
                logger.error(f"File too large: {file_size / (1024*1024):.2f} MB")
                print(f"Error: File too large ({file_size / (1024*1024):.2f} MB). Maximum size is {max_size_mb} MB.")
                return
            
            # Log the request details
            logger.info(f"Transcribing audio file: {file_path}")
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            logger.info(f"File format: {file_ext}")
            
            # Try with a different model if the default one fails
            # models_to_try = ["en-US-0.6b", "en-US-0.3b"]  # Try both models
            models_to_try = ["en-NER"]
            last_error = None
            for try_model in models_to_try:
                retry_count = 0
                max_retries = 2
                
                while retry_count <= max_retries:
                    try:
                        logger.info(f"Attempting transcription with model: {try_model} (Attempt {retry_count+1}/{max_retries+1})")
                        response = client.speech_to_text(
                            audio_file_path=file_path,
                            model_name=try_model,
                            timestamps=True,
                            boosted_lm_words=["specific", "terms"],
                            boosted_lm_score=80
                        )
                        
                        if response and hasattr(response, 'transcript'):
                            logger.info(f"Transcription successful with model: {try_model}")
                            print("Response type:", type(response))
                            print(f"Transcription: {response.transcript}")
                            
                            # Handle timestamps if available
                            if hasattr(response, 'timestamps'):
                                print("\nTimestamps:")
                                for ts in response.timestamps:
                                    print(f"Word: {ts.get('word', '')}, Start: {ts.get('startTime', 0)}, End: {ts.get('endTime', 0)}, Confidence: {ts.get('confidence', 0)}")
                            
                            # Handle diarization output if available
                            if hasattr(response, 'diarize_output') and response.diarize_output:
                                print("\nDiarization Output:")
                                for segment in response.diarize_output:
                                    if segment.get('text'):
                                        print(f"Speaker {segment.get('speaker_id', 'unknown')}: {segment.get('text', '')} ({segment.get('start_timestamp', 0)} - {segment.get('end_timestamp', 0)})")
                            
                            # Print additional metadata if available
                            if hasattr(response, 'duration_seconds'):
                                print(f"\nDuration: {response.duration_seconds} seconds")
                            if hasattr(response, 'language_code'):
                                print(f"Language: {response.language_code}")
                            
                            return
                        else:
                            last_error = "No transcription was returned from the API"
                            logger.error(f"No transcription returned from API with model {try_model}")
                            break  # No point retrying if we got a response but no text
                    except Exception as api_error:
                        error_msg = str(api_error)
                        logger.error(f"Error with model {try_model}: {error_msg}")
                        last_error = error_msg
                        
                        # Handle API errors with retries
                        error_result = handle_api_error(error_msg, "transcription", retry_count, max_retries)
                        if error_result is not None:  # If we should not retry
                            if retry_count == max_retries:  # If this was our last retry
                                break  # Try next model
                            else:
                                print(f"Error: {error_result}")
                                return
                        
                        retry_count += 1
            
            # If we get here, all models failed
            if "HTTP 500" in last_error:
                logger.error(f"All transcription attempts failed with HTTP 500: {last_error}")
                print(f"Error: Server error during transcription. This might be a temporary issue with the Whissle API. Please try again later or contact Whissle support. Error: {last_error}")
            else:
                logger.error(f"All transcription attempts failed: {last_error}")
                print(f"Error: Failed to transcribe audio: {last_error}")
                
        except Exception as e:
            logger.error(f"Unexpected error during transcription: {str(e)}")
            print(f"Error: Failed to transcribe audio: {str(e)}")
    
    def test_diarization(file_path="test.wav"):
        print("\n=== Testing Speech Diarization ===")
        logger.info(f"Testing Speech Diarization with file: {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                print(f"Error: Audio file not found: {file_path}")
                return
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"Audio file is empty: {file_path}")
                print(f"Error: Audio file is empty: {file_path}")
                return
            
            # Check file format
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
                logger.error(f"Unsupported audio format: {file_ext}")
                print(f"Error: Unsupported audio format: {file_ext}. Supported formats: wav, mp3, ogg, flac, m4a")
                return
            
            # Check file size limits (if file is too large)
            max_size_mb = 25  # Assuming a reasonable limit
            if file_size > max_size_mb * 1024 * 1024:
                logger.error(f"File too large: {file_size / (1024*1024):.2f} MB")
                print(f"Error: File too large ({file_size / (1024*1024):.2f} MB). Maximum size is {max_size_mb} MB.")
                return
            
            # Log the request details
            logger.info(f"Diarizing audio file: {file_path}")
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            logger.info(f"File format: {file_ext}")
            
            # Try with a different model if the default one fails
            # models_to_try = ["en-US-0.6b", "en-US-0.3b"]  # Try both models
            models_to_try = ["en-NER"]
            last_error = None
            for try_model in models_to_try:
                retry_count = 0
                max_retries = 2
                
                while retry_count <= max_retries:
                    try:
                        logger.info(f"Attempting diarization with model: {try_model} (Attempt {retry_count+1}/{max_retries+1})")
                        response = client.diarize_stt(
                            audio_file_path=file_path,
                            model_name=try_model,
                            max_speakers=2,
                            boosted_lm_words=["specific", "terms"],
                            boosted_lm_score=80
                        )
                        print(response)
                        if response and hasattr(response, 'diarize_output') and response.diarize_output:
                            logger.info(f"Diarization successful with model: {try_model}")
                            print("Response type:", type(response))
                            
                            # Print the full transcript if available
                            if hasattr(response, 'transcript'):
                                print(f"\nFull Transcript: {response.transcript}")
                            
                            # Print diarization segments
                            print("\nDiarized Transcription:")
                            for segment in response.diarize_output:
                                if segment.get('text'):
                                    print(f"Speaker {segment.get('speaker_id', 'unknown')}: {segment.get('text', '')} ({segment.get('start_timestamp', 0)} - {segment.get('end_timestamp', 0)})")
                            
                            # Print timestamps if available
                            if hasattr(response, 'timestamps') and response.timestamps:
                                print("\nDetailed Timestamps:")
                                for ts in response.timestamps:
                                    if ts.get('word'):
                                        print(f"Word: {ts.get('word', '')}, Start: {ts.get('startTime', 0)}, End: {ts.get('endTime', 0)}, Speaker: {ts.get('speakerTag', 'unknown')}, Confidence: {ts.get('confidence', 0)}")
                            
                            # Print additional metadata if available
                            if hasattr(response, 'duration_seconds'):
                                print(f"\nDuration: {response.duration_seconds} seconds")
                            if hasattr(response, 'language_code'):
                                print(f"Language: {response.language_code}")
                            
                            return
                        else:
                            last_error = "No diarized transcription was returned from the API"
                            logger.error(f"No diarized transcription returned from API with model {try_model}")
                            break  # No point retrying if we got a response but no segments
                    except Exception as api_error:
                        error_msg = str(api_error)
                        logger.error(f"Error with model {try_model}: {error_msg}")
                        last_error = error_msg
                        
                        # Handle API errors with retries
                        error_result = handle_api_error(error_msg, "diarization", retry_count, max_retries)
                        if error_result is not None:  # If we should not retry
                            if retry_count == max_retries:  # If this was our last retry
                                break  # Try next model
                            else:
                                print(f"Error: {error_result}")
                                return
                        
                        retry_count += 1
            
            # If we get here, all models failed
            if "HTTP 500" in last_error:
                logger.error(f"All diarization attempts failed with HTTP 500: {last_error}")
                print(f"Error: Server error during diarization. This might be a temporary issue with the Whissle API. Please try again later or contact Whissle support. Error: {last_error}")
            else:
                logger.error(f"All diarization attempts failed: {last_error}")
                print(f"Error: Failed to diarize speech: {last_error}")
                
        except Exception as e:
            logger.error(f"Unexpected error during diarization: {str(e)}")
            print(f"Error: Failed to diarize speech: {str(e)}")
    
    def test_translation(text="Hello, world!", source="en", target="es"):
        print("\n=== Testing Translation ===")
        logger.info(f"Testing Translation from {source} to {target}")
        
        try:
            if not text:
                logger.error("Empty text provided for translation")
                print("Error: Text is required")
                return
            
            # Log the request details
            logger.info(f"Translating text from {source} to {target}")
            logger.info(f"Text length: {len(text)} characters")
            
            retry_count = 0
            max_retries = 2
            
            while retry_count <= max_retries:
                try:
                    logger.info(f"Attempting translation (Attempt {retry_count+1}/{max_retries+1})")
                    print(f"Attempting to translate from {source} to {target}")
                    print(f"Original text: {text}")
                    
                    response = client.machine_translation(
                        text=text,
                        source_language=source,
                        target_language=target
                    )
                    
                    if response and response.translated_text:
                        logger.info("Translation successful")
                        print("Response type:", type(response))
                        print(f"Translated text: {response.translated_text}")
                        return
                    else:
                        logger.error("No translation was returned from the API")
                        print("Error: No translation was returned from the API")
                        return
                except Exception as api_error:
                    error_msg = str(api_error)
                    logger.error(f"Translation error: {error_msg}")
                    
                    # Handle API errors with retries
                    error_result = handle_api_error(error_msg, "translation", retry_count, max_retries)
                    if error_result is not None:  # If we should not retry
                        print(f"Error: {error_result}")
                        return
                    
                    retry_count += 1
            
            # If we get here, all retries failed
            logger.error(f"All translation attempts failed after {max_retries+1} attempts")
            print(f"Error: Failed to translate text after {max_retries+1} attempts")
        except Exception as e:
            logger.error(f"Unexpected error during translation: {str(e)}")
            print(f"Error: Failed to translate text: {str(e)}")
    
    def test_summarization(text="This is a long text that needs to be summarized."):
        print("\n=== Testing Text Summarization ===")
        logger.info("Testing Text Summarization")
        
        try:
            if not text:
                logger.error("Empty content provided for summarization")
                print("Error: Content is required")
                return
            
            # Log the request details
            logger.info(f"Summarizing text using model: openai")
            logger.info(f"Text length: {len(text)} characters")
            
            retry_count = 0
            max_retries = 2
            
            while retry_count <= max_retries:
                try:
                    logger.info(f"Attempting summarization (Attempt {retry_count+1}/{max_retries+1})")
                    print(f"Original text: {text}")
                    
                    response = client.llm_text_summarizer(
                        content=text,
                        model_name="openai",
                        instruction="Provide a brief summary"
                    )
                    logger.info("Summarization successful")
                    print("Response type:", type(response))
                    print(f"Summary: {response.response}")

                    if response and response.response:
                        logger.info("Summarization successful")
                        print("Response type:", type(response))
                        print(f"Summary: {response.response}")
                        return
                    else:
                        logger.error("No summary was returned from the API")
                        print("Error: No summary was returned from the API")
                        return
                except Exception as api_error:
                    error_msg = str(api_error)
                    logger.error(f"Summarization error: {error_msg}")
                    
                    # Handle API errors with retries
                    error_result = handle_api_error(error_msg, "summarization", retry_count, max_retries)
                    if error_result is not None:  # If we should not retry
                        print(f"Error: {error_result}")
                        return
                    
                    retry_count += 1
            
            # If we get here, all retries failed
            logger.error(f"All summarization attempts failed after {max_retries+1} attempts")
            print(f"Error: Failed to summarize text after {max_retries+1} attempts")
        except Exception as e:
            logger.error(f"Unexpected error during summarization: {str(e)}")
            print(f"Error: Failed to summarize text: {str(e)}")

    # Run all tests
    print("Starting Whissle API Tests...")
    print("=" * 50)
    logger.info("Starting Whissle API Tests")
    
    # Always run list_models as it doesn't require any files
    test_list_models()
    
    # Run translation and summarization tests
    test_translation()
    test_summarization(text="""
    The quick brown fox jumps over the lazy dog. This is a pangram, a sentence that contains 
    every letter of the English alphabet at least once. Pangrams are often used to display 
    font samples and test keyboards. The most famous pangram is probably the one used by 
    typing practice programs: "The quick brown fox jumps over the lazy dog."
    """)
    
    # Run speech-related tests if audio files exist
    if os.path.exists("test.wav"):
        test_speech_to_text()
    else:
        print("\n=== Speech to Text Test Skipped ===")
        print("test.wav not found. Place an audio file named 'test.wav' in the same directory to test speech-to-text.")
        logger.info("Speech to Text Test Skipped - test.wav not found")
    
    if os.path.exists("test.wav"):
        test_diarization()
    else:
        print("\n=== Diarization Test Skipped ===")
        print("test.wav not found. Place a multi-speaker audio file named 'test.wav' in the same directory to test diarization.")
        logger.info("Diarization Test Skipped - test.wav not found")
    
    print("\nAll tests completed!")
    print("=" * 50)
    logger.info("All tests completed")

if __name__ == "__main__":
    main()