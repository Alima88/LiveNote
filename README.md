# LiveNote

LiveNote is a powerful application designed to transcribe live audio streams, take notes, and summarize key points in real-time. Leveraging advanced AI models like "Whisper" for transcription and "Phi-3-mini-4k-instruct" for summarization, LiveNote enhances productivity and ensures you never miss important details from your conversations or meetings.

## Features

- **Real-time Transcription:** Convert live audio streams into text instantly using the Whisper model.
- **Note-taking:** Extract essential notes from the transcriptions.
- **Summarization:** Summarize main points from the transcriptions to get concise insights.

## Installation

### Docker Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/livenote.git
   cd livenote
   ```

2. Create a .env file:
    ```bash
    cp sample.env .env
    ```

3. Build and run the containers using Docker Compose:

    ```bash
    docker-compose up --build
    ```

4. Access the API:
Open your browser and go to http://127.0.0.1:8000 to access the API documentation and endpoints.

### Manual Setup

1. Set up a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows, use `env\Scripts\activate`
    ```

2. Install the dependencies:
    ```bash
    pip install -r app/ml-requirements.txt
    pip install -r app/requirements.txt
    ```

3. Set up environment variables:
    Create a .env file in the project root and add your environment variables (e.g., API keys).

4. Start the FastAPI server:

    ```bash
    python app.py
    ```

## Usage
### WebSocket API
- Endpoint: `/ws/transcribe`
    - Description: Accepts live audio streams for real-time transcription.

### REST API
- Endpoint: `/api/notes`
    - Description: Extracts notes from the provided transcription.
    - Method: `POST`
    - Request Body:
        ```json
        {
            "transcription": "Your transcribed text here"
        }
        ```
- Endpoint: `/api/summarize`
    - Description: Summarizes the main points from the provided transcription.
    - Method: `POST`
    - Request Body:
        ```json
        {
            "transcription": "Your transcribed text here"
        }
        ```

## Models Used
- [Whisper](https://github.com/openai/whisper): For live audio transcription.
- [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct): For extracting notes and summarizing main points.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your changes.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [WhisperFusion](https://github.com/collabora/WhisperFusion) - builds upon the capabilities of WhisperLive and WhisperSpeech to provide a seamless conversations with an AI.

## Contact
For any questions or feedback, please contact us at ali8molaee@gmail.com .