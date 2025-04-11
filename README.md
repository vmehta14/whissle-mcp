# Whissle MCP Server

A Python-based server that provides access to Whissle API endpoints for speech-to-text, diarization, translation, and text summarization.

## ⚠️ Important Notes

- This server provides access to Whissle API endpoints which may incur costs
- Each tool that makes an API call is marked with a cost warning
- Please follow these guidelines:
  1. Only use tools when explicitly requested by the user
  2. For tools that process audio, consider the length of the audio as it affects costs
  3. Some operations like translation or summarization may have higher costs
  4. Tools without cost warnings in their description are free to use as they only read existing data

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Whissle API authentication token

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd whissle_mcp
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -e .
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following content:
   ```
   WHISSLE_AUTH_TOKEN=insert_auth_token_here  # Replace with your actual Whissle API token
   WHISSLE_MCP_BASE_PATH=/path/to/your/base/directory
   ```
   ⚠️ **Important**: Never commit your actual token to the repository. The `.env` file is included in `.gitignore` to prevent accidental commits.

5. Configure Claude Integration:
   Copy `claude_config.example.json` to `claude_config.json` and update the paths:
   ```json
   {
       "mcpServers": {
           "Whissle": {
               "command": "/path/to/your/venv/bin/python",
               "args": [
                   "/path/to/whissle_mcp/server.py"
               ],
               "env": {
                   "WHISSLE_AUTH_TOKEN": "insert_auth_token_here"
               }
           }
       }
   }
   ```
   - Replace `/path/to/your/venv/bin/python` with the actual path to your Python interpreter in the virtual environment
   - Replace `/path/to/whissle_mcp/server.py` with the actual path to your server.py file
## Configuration

### Environment Variables

- `WHISSLE_AUTH_TOKEN`: Your Whissle API authentication token (required)
  - This is a sensitive credential that should never be shared or committed to version control
  - Contact your administrator to obtain a valid token
  - Store it securely in your local `.env` file
- `WHISSLE_MCP_BASE_PATH`: Base directory for file operations (optional, defaults to user's Desktop)

### Supported Audio Formats

The server supports the following audio formats:
- WAV (.wav)
- MP3 (.mp3)
- OGG (.ogg)
- FLAC (.flac)
- M4A (.m4a)

### File Size Limits

- Maximum file size: 25 MB
- Files larger than this limit will be rejected

## Available Tools

### 1. Speech to Text
Convert speech to text using the Whissle API.

```python
response = speech_to_text(
    audio_file_path="path/to/audio.wav",
    model_name="en-NER",  # Default model
    timestamps=True,      # Include word timestamps
    boosted_lm_words=["specific", "terms"],  # Words to boost in recognition
    boosted_lm_score=80   # Score for boosted words (0-100)
)
```

### 2. Speech Diarization
Convert speech to text with speaker identification.

```python
response = diarize_speech(
    audio_file_path="path/to/audio.wav",
    model_name="en-NER",  # Default model
    max_speakers=2,       # Maximum number of speakers to identify
    boosted_lm_words=["specific", "terms"],
    boosted_lm_score=80
)
```

### 3. Text Translation
Translate text from one language to another.

```python
response = translate_text(
    text="Hello, world!",
    source_language="en",
    target_language="es"
)
```

### 4. Text Summarization
Summarize text using an LLM model.

```python
response = summarize_text(
    content="Long text to summarize...",
    model_name="openai",  # Default model
    instruction="Provide a brief summary"  # Optional
)
```

### 5. List ASR Models
List all available ASR models and their capabilities.

```python
response = list_asr_models()
```

## Response Format

### Speech to Text and Diarization
```python
{
    "transcript": "The transcribed text",
    "duration_seconds": 10.5,
    "language_code": "en",
    "timestamps": [
        {
            "word": "The",
            "startTime": 0,
            "endTime": 100,
            "confidence": 0.95
        }
    ],
    "diarize_output": [
        {
            "text": "The transcribed text",
            "speaker_id": 1,
            "start_timestamp": 0,
            "end_timestamp": 10.5
        }
    ]
}
```

### Translation
```python
{
    "type": "text",
    "text": "Translation:\nTranslated text here"
}
```

### Summarization
```python
{
    "type": "text",
    "text": "Summary:\nSummarized text here"
}
```

### Error Response
```python
{
    "error": "Error message here"
}
```

## Error Handling

The server includes robust error handling with:
- Automatic retries for HTTP 500 errors
- Detailed error messages for different failure scenarios
- File validation (existence, size, format)
- Authentication checks

Common error types:
- HTTP 500: Server error (with retry mechanism)
- HTTP 413: File too large
- HTTP 415: Unsupported file format
- HTTP 401/403: Authentication error

## Running the Server

1. Start the server:
   ```bash
   mcp serve
   ```

2. The server will be available at the default MCP port (usually 8000)

## Testing

A test script is provided to verify the functionality of all tools:

```bash
python test_whissle.py
```

The test script will:
1. Check for authentication token
2. Test all available tools
3. Provide detailed output of each operation
4. Handle errors gracefully

## Support

For issues or questions, please:
1. Check the error messages for specific details
2. Verify your authentication token
3. Ensure your audio files meet the requirements
4. Contact Whissle support for API-related issues

## License

[Add your license information here] 