from starlette.responses import JSONResponse

from app.core.config import settings


class _RequestBodyTooLarge(Exception):
    pass


class ImageUploadLimitMiddleware:
    """Reject oversized disease upload requests before multipart parsing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        upload_path = (
            "/"
            + settings.api_v1_prefix.strip("/")
            + "/diseases/"
        )
        if scope["type"] != "http" or not scope.get("path", "").startswith(upload_path):
            await self.app(scope, receive, send)
            return

        # Allow room for multipart boundaries and headers around the image.
        max_request_bytes = settings.max_image_upload_bytes + 64 * 1024
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > max_request_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                pass

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > max_request_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope, receive, send):
        response = JSONResponse(
            status_code=413,
            content={"detail": "Request body exceeds the image upload limit."},
        )
        await response(scope, receive, send)
