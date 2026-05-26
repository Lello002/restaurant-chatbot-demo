import os
import uvicorn

port = int(os.environ.get("PORT", "10000"))

uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=port,
    reload=False
)