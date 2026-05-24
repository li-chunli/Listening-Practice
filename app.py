import asyncio
import json
import os
import re
import shutil
import sqlite3
from functools import lru_cache
from urllib.request import Request, urlopen
from urllib.parse import quote
import edge_tts
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORDS_DIR = os.path.join(BASE_DIR, "word")
DICT_DB = os.path.join(BASE_DIR, "dict", "stardict.db")


# ---- Dictionary lookup ----
class DictDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @lru_cache(maxsize=2000)
    def lookup(self, word):
        """Look up a word, return (ipa_uk, ipa_us, definition) or None."""
        cur = self.conn.cursor()
        # Try exact match first
        for variant in (word, word.lower(), word.capitalize()):
            cur.execute(
                "SELECT word, phonetic, translation, definition FROM stardict WHERE word=?",
                (variant,),
            )
            row = cur.fetchone()
            if row:
                if row["phonetic"]:
                    return self._build_result(row)
                # Found the word but no phonetic — try base forms
                base = self._find_base_with_phonetic(cur, word)
                if base:
                    return self._build_result(base, override_trans=row)

                return self._build_result(row)

        # Not found at all — try stemming
        for base_form in self._stem(word):
            for variant in (base_form, base_form.lower(), base_form.capitalize()):
                cur.execute(
                    "SELECT word, phonetic, translation, definition FROM stardict WHERE word=?",
                    (variant,),
                )
                row = cur.fetchone()
                if row:
                    return self._build_result(row)
            else:
                continue
            break

        return None

    def _find_base_with_phonetic(self, cur, word):
        """Try to find a base form of the word that has phonetics."""
        for base_form in self._stem(word):
            for variant in (base_form, base_form.lower(), base_form.capitalize()):
                cur.execute(
                    "SELECT word, phonetic, translation, definition FROM stardict WHERE word=? AND phonetic IS NOT NULL AND phonetic!=''",
                    (variant,),
                )
                row = cur.fetchone()
                if row:
                    return row
            else:
                continue
            break
        return None

    def _build_result(self, row, override_trans=None):
        ipa = self._clean_phonetic(row["phonetic"] or "")
        trans = (override_trans["translation"] or "") if override_trans else (row["translation"] or "")
        defn = (override_trans["definition"] or "") if override_trans else (row["definition"] or "")

        chinese = trans.split("\n")[0].strip() if trans else ""
        english = defn.split("\n")[0].strip() if defn else ""

        if chinese and english:
            combined = chinese + "\n" + english
        elif chinese:
            combined = chinese
        elif english:
            combined = english
        else:
            combined = ""

        return (ipa, ipa, combined)

    @staticmethod
    def _stem(word):
        """Generate possible base forms of an inflected word."""
        w = word.lower()
        candidates = []
        # Plurals / 3rd person
        if w.endswith("ies") and len(w) > 4:
            candidates.append(w[:-3] + "y")
        if w.endswith("ves") and len(w) > 4:
            candidates.append(w[:-3] + "f")
            candidates.append(w[:-3] + "fe")
        if w.endswith("es") and len(w) > 3:
            candidates.append(w[:-2])  # dishes -> dish
            candidates.append(w[:-1])  # dishes -> dishe (unlikely but harmless)
        if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
            candidates.append(w[:-1])
        # Past tense / participles
        if w.endswith("ied") and len(w) > 4:
            candidates.append(w[:-3] + "y")
        if w.endswith("ed") and len(w) > 3:
            candidates.append(w[:-2])
            candidates.append(w[:-1])  # baked -> bake
        if w.endswith("ing") and len(w) > 4:
            candidates.append(w[:-3])
            candidates.append(w[:-3] + "e")  # making -> make
        # Comparative / superlative
        if w.endswith("ier") and len(w) > 4:
            candidates.append(w[:-3] + "y")
        if w.endswith("iest") and len(w) > 5:
            candidates.append(w[:-4] + "y")
        if w.endswith("er") and len(w) > 3:
            candidates.append(w[:-2])
            candidates.append(w[:-1])
        if w.endswith("est") and len(w) > 4:
            candidates.append(w[:-3])
            candidates.append(w[:-2])
        # Adverbs
        if w.endswith("ly") and len(w) > 3:
            candidates.append(w[:-2])
        return candidates

    @staticmethod
    def _clean_phonetic(phonetic):
        """Normalize ECDICT phonetic to a cleaner display format."""
        if not phonetic:
            return "/?/"
        # Remove leading dot (syllable boundary marker)
        p = phonetic.strip()
        while p.startswith("."):
            p = p[1:]
        # Wrap in / /
        if not p.startswith("/"):
            p = "/" + p
        if not p.endswith("/"):
            p = p + "/"
        return p


# Init dictionary
dictdb = None
if os.path.exists(DICT_DB):
    dictdb = DictDB(DICT_DB)


# ---- Word parsing ----
def parse_line(line):
    """
    Parse a word line. Formats:
      word
      word|ipa_uk|ipa_us|definition   (overrides dictionary)
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split("|")
    word = parts[0].strip()
    if not word:
        return None

    ipa_uk = "/?/"
    ipa_us = "/?/"
    definition = ""

    if len(parts) >= 4:
        # Inline metadata overrides everything
        ipa_uk = parts[1].strip() or "/?/"
        ipa_us = parts[2].strip() or "/?/"
        definition = parts[3].strip()
    else:
        # Look up from local dictionary
        if dictdb:
            result = dictdb.lookup(word)
            if result:
                ipa_uk, ipa_us, definition = result

    return {
        "word": word,
        "ipa_uk": ipa_uk,
        "ipa_us": ipa_us,
        "def": definition,
    }


def load_words_from_dir(directory):
    all_words = []
    if not os.path.isdir(directory):
        return all_words
    for f in sorted(os.listdir(directory)):
        if f.endswith(".txt"):
            filepath = os.path.join(directory, f)
            with open(filepath) as fh:
                for line in fh:
                    item = parse_line(line)
                    if item:
                        all_words.append(item)
    return all_words


# ---- Directory helpers ----
def get_dir_path(dirname):
    """Return absolute path for a word sub-directory; empty string = WORDS_DIR root."""
    if not dirname:
        return WORDS_DIR
    safe = re.sub(r'[\\/\x00-\x1f\x7f]', '', dirname).replace('..', '').strip()
    if not safe:
        raise ValueError("无效目录名")
    path = os.path.normpath(os.path.join(WORDS_DIR, safe))
    if not path.startswith(os.path.abspath(WORDS_DIR) + os.sep):
        raise ValueError("路径非法")
    return path


# ---- Routes ----
@app.route("/ico/<path:filename>")
def ico_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, "ico"), filename)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dict/status")
def dict_status():
    return jsonify({
        "available": dictdb is not None,
        "db_path": DICT_DB,
    })


@app.route("/api/dirs")
def list_dirs():
    dirs = [{"name": "", "label": "默认"}]
    if os.path.isdir(WORDS_DIR):
        for item in sorted(os.listdir(WORDS_DIR)):
            full = os.path.join(WORDS_DIR, item)
            if os.path.isdir(full):
                cnt = sum(1 for x in os.listdir(full) if x.endswith(".txt"))
                dirs.append({"name": item, "label": item, "file_count": cnt})
    return jsonify(dirs)


@app.route("/api/dirs", methods=["POST"])
def create_dir():
    data = request.get_json(silent=True)
    if not data or not data.get("name"):
        return jsonify({"error": "缺少名称"}), 400
    try:
        path = get_dir_path(data["name"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if os.path.exists(path):
        return jsonify({"error": "目录已存在"}), 409
    os.makedirs(path)
    return jsonify({"name": os.path.basename(path)})


@app.route("/api/dirs/<dirname>", methods=["DELETE"])
def delete_dir(dirname):
    try:
        path = get_dir_path(dirname)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not os.path.isdir(path) or os.path.abspath(path) == os.path.abspath(WORDS_DIR):
        return jsonify({"error": "目录不存在或不可删除"}), 404
    shutil.rmtree(path)
    return jsonify({"ok": True})


@app.route("/api/files")
def list_files():
    try:
        directory = get_dir_path(request.args.get("dir", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    files = []
    if not os.path.isdir(directory):
        return jsonify([])
    for f in sorted(os.listdir(directory)):
        if f.endswith(".txt"):
            filepath = os.path.join(directory, f)
            with open(filepath) as fh:
                count = sum(1 for line in fh if parse_line(line))
            files.append({"name": f, "word_count": count})
    return jsonify(files)


@app.route("/api/words")
def get_words():
    filename = request.args.get("file", "")
    try:
        directory = get_dir_path(request.args.get("dir", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if filename:
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        source = []
        with open(filepath) as fh:
            for line in fh:
                item = parse_line(line)
                if item:
                    source.append(item)
    else:
        source = load_words_from_dir(directory)

    return jsonify(source)


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not f.filename.endswith(".txt"):
        return jsonify({"error": "Only .txt files allowed"}), 400
    try:
        directory = get_dir_path(request.args.get("dir", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f.filename)
    f.save(filepath)
    with open(filepath) as fh:
        count = sum(1 for line in fh if parse_line(line))
    return jsonify({"name": f.filename, "word_count": count})


@app.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    try:
        directory = get_dir_path(request.args.get("dir", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    os.remove(filepath)
    return jsonify({"ok": True})


@app.route("/api/word", methods=["DELETE"])
def delete_word():
    filename = request.args.get("file", "")
    word_to_delete = request.args.get("word", "")
    try:
        directory = get_dir_path(request.args.get("dir", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not filename or not word_to_delete:
        return jsonify({"error": "缺少参数"}), 400
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "文件不存在"}), 404
    with open(filepath, encoding="utf-8") as fh:
        lines = fh.readlines()
    new_lines = []
    removed = 0
    for line in lines:
        item = parse_line(line)
        if item and item["word"].lower() == word_to_delete.lower():
            removed += 1
        else:
            new_lines.append(line)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.writelines(new_lines)
    return jsonify({"removed": removed})


@app.route("/api/wrongwords", methods=["POST"])
def add_wrong_word():
    data = request.get_json(silent=True)
    if not data or not data.get("word") or not data.get("file"):
        return jsonify({"error": "缺少参数"}), 400
    try:
        directory = get_dir_path(data.get("dir", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    word = data["word"].strip()
    if not word:
        return jsonify({"error": "单词为空"}), 400
    wrong_file = os.path.join(directory, "错词表.txt")
    existing = set()
    if os.path.exists(wrong_file):
        with open(wrong_file, encoding="utf-8") as fh:
            for line in fh:
                item = parse_line(line)
                if item:
                    existing.add(item["word"].lower())
    if word.lower() not in existing:
        with open(wrong_file, "a", encoding="utf-8") as fh:
            fh.write(word + "\n")
    return jsonify({"ok": True})



def _valid_mp3(data):
    """Check data looks like a complete MP3: valid header + minimum size."""
    if len(data) < 2048:
        return False
    return data[:3] == b'ID3' or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)


@lru_cache(maxsize=2000)
def _fetch_google(word, lang):
    url = "https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=" + lang + "&q=" + quote(word)
    for _ in range(3):
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urlopen(req, timeout=10).read()
        if _valid_mp3(data):
            return data
    raise ValueError(f"truncated audio for '{word}'")


@lru_cache(maxsize=2000)
def _fetch_baidu(word, lang):
    url = "https://fanyi.baidu.com/gettts?lan=" + lang + "&text=" + quote(word) + "&spd=3"
    for _ in range(3):
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urlopen(req, timeout=10).read()
        if _valid_mp3(data):
            return data
    raise ValueError(f"truncated audio for '{word}'")


@app.route("/api/tts/google")
def tts_google():
    word = request.args.get("word", "")
    lang = request.args.get("lang", "en")
    if not word:
        return jsonify({"error": "Missing word"}), 400
    try:
        return app.response_class(_fetch_google(word, lang), mimetype="audio/mpeg")
    except Exception as e:
        _fetch_google.cache_clear()
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts/baidu")
def tts_baidu():
    word = request.args.get("word", "")
    lang = request.args.get("lang", "en")
    if not word:
        return jsonify({"error": "Missing word"}), 400
    try:
        return app.response_class(_fetch_baidu(word, lang), mimetype="audio/mpeg")
    except Exception as e:
        _fetch_baidu.cache_clear()
        return jsonify({"error": str(e)}), 500


_EDGE_VOICES = {
    ("en-GB", "female"): "en-GB-SoniaNeural",
    ("en-GB", "male"):   "en-GB-RyanNeural",
    ("en-US", "female"): "en-US-JennyNeural",
    ("en-US", "male"):   "en-US-GuyNeural",
    ("en-AU", "female"): "en-AU-NatashaNeural",
    ("en-AU", "male"):   "en-AU-WilliamNeural",
    ("en-IN", "female"): "en-IN-NeerjaNeural",
    ("en-IN", "male"):   "en-IN-PrabhatNeural",
    ("en-CA", "female"): "en-CA-ClaraNeural",
    ("en-CA", "male"):   "en-CA-LiamNeural",
    ("en-IE", "female"): "en-IE-EmilyNeural",
    ("en-IE", "male"):   "en-IE-ConnorNeural",
}


@lru_cache(maxsize=2000)
def _fetch_edge(word, voice):
    async def _generate():
        communicate = edge_tts.Communicate(word, voice)
        buf = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        return buf
    for _ in range(3):
        data = asyncio.run(_generate())
        if _valid_mp3(data):
            return data
    raise ValueError(f"truncated audio for '{word}'")


@app.route("/api/tts/edge")
def tts_edge():
    word = request.args.get("word", "")
    lang = request.args.get("lang", "en-GB")
    gender = request.args.get("gender", "female")
    if not word:
        return jsonify({"error": "Missing word"}), 400
    voice = _EDGE_VOICES.get((lang, gender), "en-GB-SoniaNeural")
    try:
        return app.response_class(_fetch_edge(word, voice), mimetype="audio/mpeg")
    except Exception as e:
        _fetch_edge.cache_clear()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Dict DB: {'found' if os.path.exists(DICT_DB) else 'NOT FOUND'} at {DICT_DB}")
    app.run(host="0.0.0.0", port=8080, debug=True)
