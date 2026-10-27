# Image Detection

Image Detection is a small full-stack image detection app. Flask serves the frontend and a JSON API; YOLO11 small analyzes images on the server.

## Run locally

1. Install Python 3.10 or newer.
2. From this folder, create a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Start the server:

   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```

4. Open <http://127.0.0.1:5000>.

The app uses the included `yolo11s.pt` pretrained COCO weights. Set `YOLO_MODEL` to a different Ultralytics model path/name to use another model. Inference runs on CPU at a 1280-pixel image size with a 5% minimum model confidence and up to 300 detections per inference view. The UI initially shows results at 25% confidence; lower the slider to reveal less-certain detections or raise it to reduce false positives. Large images are also analyzed in overlapping 1024-pixel tiles to improve detection of small objects; duplicate boxes for the same class are suppressed. Tiled inference takes longer on CPU. The default COCO model recognizes 80 trained object classes, so it cannot label classes outside its training data or guarantee that every visible object is found.

## API

- `GET /api/health` returns the API status and configured model name.
- `GET /api/system-status` returns CPU, memory, storage, and network usage for the server host. Network throughput is measured between requests, so the first sample reports no rate.
- `POST /api/detect` accepts a multipart form field named `image` with a JPG, PNG, or WEBP image up to 15 MB. It returns the image dimensions and detections with class, confidence score, and `[x, y, width, height]` pixel bounding boxes.

Images are processed in memory and are not saved by the app. They are sent to whichever server is hosting this application. Do not upload sensitive images to a server you do not control.
System status reports the hosting server's resource usage. Keep the service on localhost or a trusted private network; exposing it publicly also exposes host telemetry unless access is protected.

## Camera

On the website, select **Use camera**, allow camera access in your browser, then capture a photo. The photo is handled like an uploaded image and is sent to the app server only when you run detection. Camera access requires a secure browser context: use `http://127.0.0.1:5000` locally or deploy behind HTTPS.
