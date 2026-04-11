# Entry point — imports the Flask app from inference.py
from inference import app  # noqa: F401

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
