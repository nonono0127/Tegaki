import base64
import io
import json
import os
import re

from flask import Flask, render_template, request, jsonify, send_file
import anthropic
from dotenv import load_dotenv
from openpyxl import Workbook
from PIL import Image

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


MAX_IMAGE_PIXELS = 7000


def resize_image(image_data: bytes, media_type: str) -> tuple[bytes, str]:
    """画像が8000px超の場合にリサイズする"""
    img = Image.open(io.BytesIO(image_data))
    w, h = img.size
    if max(w, h) > MAX_IMAGE_PIXELS:
        scale = MAX_IMAGE_PIXELS / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out = io.BytesIO()
    fmt = "JPEG" if media_type in ("image/jpeg",) else img.format or "PNG"
    img.save(out, format=fmt)
    return out.getvalue(), media_type


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _call_claude(messages: list, max_tokens: int = 8192) -> str:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=max_tokens,
        messages=messages,
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def _parse_json_response(text: str):
    """JSON応答をパース（コードブロック付きも対応）"""
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    return json.loads(text)


def _encode_image(image_data: bytes, media_type: str) -> dict:
    encoded = base64.standard_b64encode(image_data).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}}


def extract_headers(image_data: bytes, media_type: str) -> list[str]:
    """テンプレート画像からフィールド名・見出しを抽出"""
    text = _call_claude(
        [
            {
                "role": "user",
                "content": [
                    _encode_image(image_data, media_type),
                    {
                        "type": "text",
                        "text": (
                            "この画像は記入式フォームのテンプレートです。"
                            "印刷されたフィールド名・項目名・見出しをすべて記載順に抽出してください。"
                            "以下のJSON形式のみを出力してください（説明・補足不要）:\n"
                            '{"headers": ["項目1", "項目2", ...]}'
                        ),
                    },
                ],
            }
        ],
        max_tokens=1024,
    )
    data = _parse_json_response(text)
    return data.get("headers", [])


def transcribe_with_headers(image_data: bytes, media_type: str, headers: list[str]) -> dict:
    """手書き画像からヘッダーに対応する内容を抽出"""
    headers_json = json.dumps(headers, ensure_ascii=False)
    empty = json.dumps({h: "" for h in headers}, ensure_ascii=False)
    text = _call_claude(
        [
            {
                "role": "user",
                "content": [
                    _encode_image(image_data, media_type),
                    {
                        "type": "text",
                        "text": (
                            f"この画像には次の項目を持つフォームが手書きで記入されています。\n"
                            f"項目: {headers_json}\n"
                            "各項目に書かれた手書きの内容を正確に文字起こしして、"
                            "以下のJSON形式のみを出力してください（説明・補足不要）:\n"
                            + empty
                        ),
                    },
                ],
            }
        ],
    )
    return _parse_json_response(text)


def transcribe_handwriting(image_data: bytes, media_type: str) -> str:
    return _call_claude(
        [
            {
                "role": "user",
                "content": [
                    _encode_image(image_data, media_type),
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
    )


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
    image_data, media_type = resize_image(file.read(), media_type)

    try:
        result = transcribe_handwriting(image_data, media_type)
        return jsonify({"text": result})
    except anthropic.AuthenticationError:
        return jsonify({"error": "APIキーが無効です。ANTHROPIC_API_KEY を確認してください"}), 500
    except anthropic.APIConnectionError:
        return jsonify({"error": "APIへの接続に失敗しました。ネットワークを確認してください"}), 503
    except Exception as e:
        return jsonify({"error": f"文字起こし中にエラーが発生しました: {str(e)}"}), 500


@app.route("/transcribe-excel", methods=["POST"])
def transcribe_excel():
    if "template" not in request.files or "handwriting" not in request.files:
        return jsonify({"error": "テンプレート画像と手書き画像の両方が必要です"}), 400

    template_file = request.files["template"]
    handwriting_file = request.files["handwriting"]

    if not allowed_file(template_file.filename) or not allowed_file(handwriting_file.filename):
        return jsonify({"error": "対応していないファイル形式です (PNG, JPG, GIF, WebP のみ)"}), 400

    template_data, template_media = resize_image(
        template_file.read(),
        MEDIA_TYPE_MAP[template_file.filename.rsplit(".", 1)[1].lower()],
    )
    handwriting_data, hw_media = resize_image(
        handwriting_file.read(),
        MEDIA_TYPE_MAP[handwriting_file.filename.rsplit(".", 1)[1].lower()],
    )

    try:
        headers = extract_headers(template_data, template_media)
        if not headers:
            return jsonify({"error": "テンプレートから見出しを抽出できませんでした"}), 500

        content = transcribe_with_headers(handwriting_data, hw_media, headers)

        wb = Workbook()
        ws = wb.active
        ws.title = "文字起こし結果"
        ws.append(headers)
        ws.append([content.get(h, "") for h in headers])

        excel_io = io.BytesIO()
        wb.save(excel_io)
        excel_io.seek(0)

        return send_file(
            excel_io,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="tegaki_result.xlsx",
        )
    except anthropic.AuthenticationError:
        return jsonify({"error": "APIキーが無効です。ANTHROPIC_API_KEY を確認してください"}), 500
    except anthropic.APIConnectionError:
        return jsonify({"error": "APIへの接続に失敗しました。ネットワークを確認してください"}), 503
    except Exception as e:
        return jsonify({"error": f"処理中にエラーが発生しました: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
