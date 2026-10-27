import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import app as detector_app


def png_bytes():
    image = Image.new("RGB", (32, 24), color="green")
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


class DetectionApiTests(unittest.TestCase):
    def setUp(self):
        detector_app.app.config["TESTING"] = True
        self.client = detector_app.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        self.assertEqual(response.json["model"], detector_app.MODEL_NAME)

    def test_system_status_reports_host_metrics_without_exposing_paths(self):
        memory = SimpleNamespace(percent=62.5, used=5_000, total=8_000)
        storage = SimpleNamespace(percent=70.0, used=7_000, total=10_000, free=3_000)
        network = SimpleNamespace(bytes_recv=1_000, bytes_sent=500)
        with (
            patch.object(detector_app.psutil, "cpu_percent", return_value=34.5),
            patch.object(detector_app.psutil, "virtual_memory", return_value=memory),
            patch.object(detector_app.psutil, "disk_usage", return_value=storage) as disk_usage,
            patch.object(detector_app.psutil, "net_io_counters", return_value=network),
            patch.object(detector_app.time, "monotonic", return_value=10.0),
            patch.object(detector_app, "_previous_network_sample", None),
        ):
            response = self.client.get("/api/system-status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["cpu_percent"], 34.5)
        self.assertEqual(response.json["memory"], {
            "percent": 62.5, "used_bytes": 5_000, "total_bytes": 8_000,
        })
        self.assertEqual(response.json["storage"], {
            "percent": 70.0, "used_bytes": 7_000, "total_bytes": 10_000, "free_bytes": 3_000,
        })
        self.assertEqual(response.json["network"], {
            "received_bytes_per_second": None, "sent_bytes_per_second": None,
        })
        self.assertNotIn("path", response.json)
        disk_usage.assert_called_once()

    def test_system_status_reports_network_rates_after_baseline(self):
        first_network = SimpleNamespace(bytes_recv=1_000, bytes_sent=500)
        second_network = SimpleNamespace(bytes_recv=3_500, bytes_sent=1_000)
        with (
            patch.object(detector_app.psutil, "cpu_percent", return_value=10.0),
            patch.object(detector_app.psutil, "virtual_memory", return_value=SimpleNamespace(
                percent=50.0, used=500, total=1_000,
            )),
            patch.object(detector_app.psutil, "disk_usage", return_value=SimpleNamespace(
                percent=50.0, used=500, total=1_000, free=500,
            )),
            patch.object(
                detector_app.psutil, "net_io_counters",
                side_effect=[first_network, second_network],
            ),
            patch.object(detector_app.time, "monotonic", side_effect=[10.0, 15.0]),
            patch.object(detector_app, "_previous_network_sample", None),
        ):
            first_response = self.client.get("/api/system-status")
            second_response = self.client.get("/api/system-status")

        self.assertIsNone(first_response.json["network"]["received_bytes_per_second"])
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(second_response.json["network"], {
            "received_bytes_per_second": 500.0, "sent_bytes_per_second": 100.0,
        })

    def test_system_status_reports_metric_failures(self):
        with patch.object(detector_app.psutil, "virtual_memory", side_effect=detector_app.psutil.AccessDenied):
            response = self.client.get("/api/system-status")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error"], "System metrics are currently unavailable.")

    def test_frontend_is_served_by_flask(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"YOLO11 small", response.data)
        self.assertIn(b'value="25"', response.data)
        self.assertIn(b"System status", response.data)
        self.assertIn(b"CPU usage", response.data)
        response.close()

    def test_detection_requires_an_image(self):
        response = self.client.post("/api/detect")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_detection_rejects_unsupported_media_type(self):
        response = self.client.post(
            "/api/detect",
            data={"image": (io.BytesIO(b"plain text"), "note.txt", "text/plain")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("JPG, PNG, or WEBP", response.json["error"])

    def test_detection_rejects_invalid_image_bytes(self):
        response = self.client.post(
            "/api/detect",
            data={"image": (io.BytesIO(b"not an image"), "fake.png", "image/png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid", response.json["error"])

    def test_detection_returns_predictions(self):
        class BoxValue:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

            def tolist(self):
                return self.value

        class Box:
            cls = [BoxValue(0)]
            conf = [BoxValue(0.91)]
            xyxy = [BoxValue([2, 3, 18, 15])]

        class Result:
            names = {0: "person"}
            boxes = [Box()]

        class Model:
            def predict(self, **_kwargs):
                self.options = _kwargs
                return [Result()]

        model = Model()
        with patch.object(detector_app, "get_model", return_value=model):
            response = self.client.post(
                "/api/detect",
                data={"image": (io.BytesIO(png_bytes()), "sample.png", "image/png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["width"], 32)
        self.assertEqual(response.json["height"], 24)
        self.assertEqual(model.options["conf"], 0.05)
        self.assertEqual(model.options["max_det"], 300)
        self.assertEqual(model.options["imgsz"], 1280)
        self.assertEqual(response.json["predictions"], [{
            "class": "person",
            "score": 0.91,
            "bbox": [2.0, 3.0, 16.0, 12.0],
        }])
        self.assertNotIn("_source", response.json["predictions"][0])

    def test_detection_composites_transparent_images_on_white(self):
        class Result:
            names = {}
            boxes = None

        class Model:
            def predict(self, **kwargs):
                self.image = kwargs["source"]
                return [Result()]

        image_bytes = io.BytesIO()
        Image.new("RGBA", (12, 10), (0, 0, 0, 0)).save(image_bytes, format="PNG")
        image_bytes.seek(0)
        model = Model()

        with patch.object(detector_app, "get_model", return_value=model):
            response = self.client.post(
                "/api/detect",
                data={"image": (image_bytes, "transparent.png", "image/png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(model.image.mode, "RGB")
        self.assertEqual(model.image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(response.json["predictions"], [])

    def test_large_image_runs_overlapping_tiles_and_suppresses_duplicate_boxes(self):
        class BoxValue:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

            def tolist(self):
                return self.value

        class Box:
            def __init__(self, score, coords):
                self.cls = [BoxValue(0)]
                self.conf = [BoxValue(score)]
                self.xyxy = [BoxValue(coords)]

        class Result:
            names = {0: "person"}

            def __init__(self, boxes):
                self.boxes = boxes

        class Model:
            def __init__(self):
                self.calls = []

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) <= 2:
                    return [Result([Box(0.9, [100, 100, 220, 300])])]
                return [Result([])]

        model = Model()
        large_image = io.BytesIO()
        Image.new("RGB", (1400, 1200), color="green").save(large_image, format="PNG")
        large_image.seek(0)

        with patch.object(detector_app, "get_model", return_value=model):
            response = self.client.post(
                "/api/detect",
                data={"image": (large_image, "large.png", "image/png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(model.calls), 1)
        self.assertTrue(all(call["conf"] == 0.05 for call in model.calls))
        self.assertTrue(all(call["max_det"] == 300 for call in model.calls))
        self.assertEqual(len(response.json["predictions"]), 1)
        self.assertEqual(response.json["predictions"][0]["bbox"], [100.0, 100.0, 120.0, 200.0])

    def test_tile_positions_cover_image_edges_without_duplicates(self):
        self.assertEqual(detector_app._tile_positions(800), [0])
        positions = detector_app._tile_positions(2400)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1] + detector_app.TILE_SIZE, 2400)
        self.assertEqual(len(positions), len(set(positions)))

    def test_duplicate_suppression_removes_overlapping_same_class_boxes(self):
        detections = [
            {"class": "person", "score": 0.9, "bbox": [100, 100, 100, 200], "_source": "full"},
            {"class": "person", "score": 0.4, "bbox": [110, 105, 180, 190], "_source": (0, 0)},
            {"class": "bus", "score": 0.8, "bbox": [100, 100, 100, 200], "_source": "full"},
        ]

        kept = detector_app._suppress_duplicate_detections(detections)

        self.assertEqual(len(kept), 2)
        self.assertIn("bus", [item["class"] for item in kept])
        self.assertEqual(sum(item["class"] == "person" for item in kept), 1)


if __name__ == "__main__":
    unittest.main()
