import os

from expense_audit_env.server.app import app

__all__ = ["app", "main"]


def main() -> None:
    uvicorn_host = "0.0.0.0"
    uvicorn_port = int(os.environ.get("PORT", "7860"))
    import uvicorn

    uvicorn.run(app, host=uvicorn_host, port=uvicorn_port)


if __name__ == "__main__":
    main()
