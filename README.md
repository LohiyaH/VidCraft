# VidCraft

## Description
This backend application is built using FastAPI and is designed to generate videos based on user-defined topics and backgrounds. It integrates multiple AI services for text generation, audio synthesis, image creation, and subtitle generation. The application handles various tasks, including cleaning text, formatting subtitles, and generating video files using FFmpeg.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/LohiyaH/VidCraft
   cd shorty
   ```
2. Install the required dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```
   GEMINI_API_KEY=<your-gemini-api-key>
   ELEVENLABS_API_KEY=<your-elevenlabs-api-key>
   HUGGINGFACE_API_KEY=<your-huggingface-api-key>
   ```

## Usage
1. Run the FastAPI application:
   ```bash
   uvicorn backend.main:app --reload
   ```
2. Access the API documentation at `http://localhost:8000/docs`.

## API Endpoints
- **POST** `/api/generate-video`
  - **Request Body**:
    ```json
    {
      "topic": "string",
      "background": "string"
    }
    ```
  - **Response**:
    ```json
    {
      "videoUrl": "string"
    }
    ```

- **GET** `/api/test`
  - Returns a simple test message.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
