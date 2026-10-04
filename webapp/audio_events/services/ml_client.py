"""ML inference microservice HTTP client module."""

import json
import os
import types
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union
from urllib.error import HTTPError, URLError
import urllib.request
from django.conf import settings

_requests: Optional[types.ModuleType]
try:
    import requests as _requests  # type: ignore[assignment]
except ImportError:
    _requests = None
requests = _requests

_httpx: Optional[types.ModuleType]
try:
    import httpx as _httpx  # type: ignore[assignment]
except ImportError:
    _httpx = None
httpx = _httpx


class MLServiceError(Exception):
    """Base exception for ML service errors or non-200 responses."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class MLServiceUnavailable(MLServiceError):
    """Exception raised when the ML inference service is unreachable or timing out."""

    pass


def get_ml_service_url() -> str:
    """Return configured ML service predict endpoint URL."""
    url = getattr(settings, "ML_SERVICE_URL", None)
    if not url:
        url = os.getenv("ML_SERVICE_URL", "http://127.0.0.1:8001/predict")
    return url


def predict_audio_file(
    file_input: Union[str, Path, bytes, BinaryIO],
    filename: str = "audio.wav",
    service_url: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """POST audio file payload to ML inference microservice /predict endpoint.

    Args:
        file_input: File path, raw bytes, or file-like object.
        filename: Filename string for multipart upload header.
        service_url: Optional service URL override.
        timeout: HTTP request timeout in seconds.

    Returns:
        Parsed dictionary response matching PredictResponse contract.

    Raises:
        MLServiceUnavailable: If ML service connection fails or times out.
        MLServiceError: If ML service returns a non-200 HTTP response.
    """
    url = service_url or get_ml_service_url()

    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        if not p.is_file():
            raise MLServiceError(f"Audio file not found: {file_input}")
        filename = p.name
        with open(p, "rb") as f:
            payload = f.read()
    elif isinstance(file_input, bytes):
        payload = file_input
    elif hasattr(file_input, "read"):
        payload = file_input.read()
        if hasattr(file_input, "seek"):
            file_input.seek(0)
    else:
        raise MLServiceError("Invalid audio file input type provided.")

    if httpx is not None:
        try:
            files = {"file": (filename, payload, "audio/wav")}
            resp = httpx.post(url, files=files, timeout=timeout)
            if resp.status_code != 200:
                detail = resp.text
                try:
                    err_j = resp.json()
                    if isinstance(err_j, dict) and "detail" in err_j:
                        detail = str(err_j["detail"])
                except Exception:
                    pass
                raise MLServiceError(
                    f"ML service returned non-200 status code [{resp.status_code}]: {detail}",
                    status_code=resp.status_code,
                    details=detail,
                )
            return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise MLServiceUnavailable(f"ML inference service unavailable at {url}: {str(exc)}") from exc
        except MLServiceError:
            raise
        except Exception as exc:
            raise MLServiceUnavailable(f"Failed to communicate with ML service: {str(exc)}") from exc

    if requests is not None:
        try:
            files = {"file": (filename, payload, "audio/wav")}
            resp = requests.post(url, files=files, timeout=timeout)
            if resp.status_code != 200:
                detail = resp.text
                try:
                    err_j = resp.json()
                    if isinstance(err_j, dict) and "detail" in err_j:
                        detail = str(err_j["detail"])
                except Exception:
                    pass
                raise MLServiceError(
                    f"ML service returned non-200 status code [{resp.status_code}]: {detail}",
                    status_code=resp.status_code,
                    details=detail,
                )
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            raise MLServiceUnavailable(f"ML inference service unavailable at {url}: {str(exc)}") from exc
        except MLServiceError:
            raise
        except Exception as exc:
            raise MLServiceUnavailable(f"Failed to communicate with ML service: {str(exc)}") from exc

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body.extend(b"Content-Type: audio/wav\r\n\r\n")
    body.extend(payload)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read().decode("utf-8")
            return json.loads(resp_body)
    except HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="replace")
        try:
            err_j = json.loads(err_text)
            if isinstance(err_j, dict) and "detail" in err_j:
                err_text = str(err_j["detail"])
        except Exception:
            pass
        raise MLServiceError(
            f"ML service returned non-200 status code [{exc.code}]: {err_text}",
            status_code=exc.code,
            details=err_text,
        ) from exc
    except URLError as exc:
        raise MLServiceUnavailable(f"ML inference service unavailable at {url}: {str(exc)}") from exc
    except Exception as exc:
        raise MLServiceUnavailable(f"Failed to communicate with ML service: {str(exc)}") from exc
