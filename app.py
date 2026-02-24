import base64
import os
from flask import Flask, render_template, request, jsonify
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
MEDIA_TYPE_MAP = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def transcribe_handwriting(image_data: bytes, media_type: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    encoded = base64.standard_b64encode(image_data).decode("utf-8")

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "この画像に書かれている手書きの文字をすべて文字起こししてください。"
                            "文字起こし結果のみを出力し、説明や補足は不要です。"
                            "改行や段落構造など、元の文章のレイアウトをできる限り再現してください。"
                        ),
                    },
                ],
            }
        ],
    ) as stream:
        final = stream.get_final_message()

    for block in final.content:
        if block.type == "text":
            return block.text

    return ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "image" not in request.files:
        return jsonify({"error": "画像ファイルが選択されていません"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "ファイルが選択されていません"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "対応していないファイル形式です (PNG, JPG, GIF, WebP のみ)"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    media_type = MEDIA_TYPE_MAP[ext]
    image_data = file.read()

    try:
        result = transcribe_handwriting(image_data, media_type)
        return jsonify({"text": result})
    except anthropic.AuthenticationError:
        return jsonify({"error": "APIキーが無効です。ANTHROPIC_API_KEY を確認してください"}), 500
    except anthropic.APIConnectionError:
        return jsonify({"error": "APIへの接続に失敗しました。ネットワークを確認してください"}), 503
    except Exception as e:
        return jsonify({"error": f"文字起こし中にエラーが発生しました: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
