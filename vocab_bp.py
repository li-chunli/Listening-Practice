import asyncio
import os
import re
import sqlite3
from functools import lru_cache
from urllib.request import Request, urlopen
from urllib.parse import quote
import edge_tts
from flask import Blueprint, render_template, jsonify, request, send_file

vocab_bp = Blueprint("vocab", __name__, url_prefix="/vocab")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VOCAB_DIR   = os.path.join(BASE_DIR, "word", "雅思真经词汇")
DICT_VOCAB  = os.path.join(BASE_DIR, "dict", "雅思真经词汇")
VOCAB_TXT   = os.path.join(DICT_VOCAB, "vocabulary.txt")
AUDIO_BASE  = os.path.join(DICT_VOCAB, "audio")
DICT_DB     = os.path.join(BASE_DIR, "dict", "stardict.db")


# ---- Utilities (self-contained copy) ----

def safe_filename(name):
    if not name:
        return ""
    name = re.sub(r'[\x00-\x1f\x7f]', '', name)
    name = name.replace('/', '').replace('\\', '')
    name = name.replace('..', '')
    return name.strip()


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
        cur = self.conn.cursor()
        for variant in (word, word.lower(), word.capitalize()):
            cur.execute(
                "SELECT word, phonetic, translation, definition FROM stardict WHERE word=?",
                (variant,),
            )
            row = cur.fetchone()
            if row:
                if row["phonetic"]:
                    return self._build_result(row)
                base = self._find_base_with_phonetic(cur, word)
                if base:
                    return self._build_result(base, override_trans=row)
                return self._build_result(row)
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
        ipa  = self._clean_phonetic(row["phonetic"] or "")
        trans = (override_trans["translation"] or "") if override_trans else (row["translation"] or "")
        defn  = (override_trans["definition"]   or "") if override_trans else (row["definition"]   or "")
        chinese = trans.strip() if trans else ""
        english = defn.split("\n")[0].strip() if defn else ""
        if chinese:
            combined = chinese
        elif english:
            combined = english
        else:
            combined = ""
        return (ipa, ipa, combined)

    @staticmethod
    def _stem(word):
        w = word.lower()
        c = []
        if w.endswith("ies") and len(w) > 4: c.append(w[:-3] + "y")
        if w.endswith("ves") and len(w) > 4: c += [w[:-3]+"f", w[:-3]+"fe"]
        if w.endswith("es")  and len(w) > 3: c += [w[:-2], w[:-1]]
        if w.endswith("s")   and not w.endswith("ss") and len(w) > 2: c.append(w[:-1])
        if w.endswith("ied") and len(w) > 4: c.append(w[:-3]+"y")
        if w.endswith("ed")  and len(w) > 3: c += [w[:-2], w[:-1]]
        if w.endswith("ing") and len(w) > 4: c += [w[:-3], w[:-3]+"e"]
        if w.endswith("ier") and len(w) > 4: c.append(w[:-3]+"y")
        if w.endswith("iest") and len(w) > 5: c.append(w[:-4]+"y")
        if w.endswith("er")  and len(w) > 3: c += [w[:-2], w[:-1]]
        if w.endswith("est") and len(w) > 4: c += [w[:-3], w[:-2]]
        if w.endswith("ly")  and len(w) > 3: c.append(w[:-2])
        return c

    @staticmethod
    def _clean_phonetic(phonetic):
        if not phonetic:
            return "/?/"
        p = phonetic.strip()
        while p.startswith("."):
            p = p[1:]
        if not p.startswith("/"):
            p = "/" + p
        if not p.endswith("/"):
            p = p + "/"
        return p


dictdb = DictDB(DICT_DB) if os.path.exists(DICT_DB) else None


# ---- vocabulary.txt lookup dict ----
# word (lowercase) -> "pos. 中文释义\nEnglish example"

_vocab_dict = {}

def _load_vocab_txt():
    if not os.path.exists(VOCAB_TXT):
        return
    with open(VOCAB_TXT, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            # first form if "word1/word2"
            word = parts[0].strip().split("/")[0].strip()
            pos  = parts[1].strip()
            zh   = parts[2].strip()
            en   = parts[3].strip() if len(parts) >= 4 else ""
            if not word or not zh:
                continue
            pos_clean = pos.rstrip(".")
            defn = (pos_clean + ". " + zh) if pos_clean else zh
            if en:
                defn += "\n" + en
            _vocab_dict[word.lower()] = defn

_load_vocab_txt()


# ---- TTS helpers ----

_ALLOWED_LANGS = {
    "en-GB", "en-US", "en-AU", "en-IN", "en-CA", "en-IE",
    "uk", "en", "au",
}
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

def _valid_mp3(data):
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

@lru_cache(maxsize=2000)
def _fetch_edge(word, voice):
    async def _run():
        chunks = []
        async for chunk in edge_tts.Communicate(word, voice).__aiter__():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        data = b"".join(chunks)
        if _valid_mp3(data):
            return data
        raise ValueError(f"empty audio for '{word}'")
    return asyncio.run(_run())


# ---- Routes ----

@vocab_bp.route("/")
def index():
    return render_template("vocab.html")



@vocab_bp.route("/api/files")
def list_files():
    if not os.path.isdir(VOCAB_DIR):
        return jsonify([])
    files = []
    for f in sorted(os.listdir(VOCAB_DIR)):
        if f.endswith(".txt"):
            filepath = os.path.join(VOCAB_DIR, f)
            with open(filepath, encoding="utf-8") as fh:
                count = sum(1 for line in fh if line.strip() and not line.strip().startswith("#"))
            files.append({"name": f, "word_count": count})
    return jsonify(files)


@vocab_bp.route("/api/words")
def get_words():
    filename = safe_filename(request.args.get("file", ""))
    if not filename or not filename.endswith(".txt"):
        return jsonify({"error": "Invalid file"}), 400
    filepath = os.path.join(VOCAB_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    result = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            word = line.strip()
            if not word or word.startswith("#"):
                continue
            ipa = "/?/"
            if dictdb:
                r = dictdb.lookup(word)
                if r:
                    ipa = r[0]
            defn = _vocab_dict.get(word.lower(), "")
            result.append({"word": word, "ipa_uk": ipa, "ipa_us": ipa, "def": defn})
    return jsonify(result)


@vocab_bp.route("/api/word", methods=["DELETE"])
def delete_word():
    filename = safe_filename(request.args.get("file", ""))
    word_to_delete = request.args.get("word", "").strip()
    if not filename or not word_to_delete:
        return jsonify({"error": "缺少参数"}), 400
    filepath = os.path.join(VOCAB_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    with open(filepath, encoding="utf-8") as fh:
        lines = fh.readlines()
    new_lines = [l for l in lines if l.strip().lower() != word_to_delete.lower()]
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.writelines(new_lines)
    return jsonify({"removed": len(lines) - len(new_lines)})


@vocab_bp.route("/api/wrongwords", methods=["POST"])
def add_wrong_word():
    data = request.get_json(silent=True)
    if not data or not data.get("word"):
        return jsonify({"error": "缺少参数"}), 400
    word = data["word"].strip()
    if not word:
        return jsonify({"error": "单词为空"}), 400
    # Derive category from file name, e.g. "01_自然地理.txt" -> "01_自然地理"
    filename = safe_filename(data.get("file", ""))
    category = filename[:-4] if filename.endswith(".txt") else ""
    cat_dir = os.path.join(VOCAB_DIR, category) if category else VOCAB_DIR
    if not os.path.isdir(cat_dir):
        cat_dir = VOCAB_DIR
    wrong_file = os.path.join(cat_dir, "错词表.txt")
    existing = set()
    if os.path.exists(wrong_file):
        with open(wrong_file, encoding="utf-8") as fh:
            for line in fh:
                w = line.strip()
                if w:
                    existing.add(w.lower())
    if word.lower() not in existing:
        with open(wrong_file, "a", encoding="utf-8") as fh:
            fh.write(word + "\n")
    return jsonify({"ok": True})


@vocab_bp.route("/api/tts/local")
def tts_local():
    word     = request.args.get("word", "").strip()
    category = safe_filename(request.args.get("category", "").strip())
    if not word or not category:
        return jsonify({"error": "Missing word or category"}), 400
    # Try exact filename match first, then lowercase
    for name in (word, word.lower()):
        audio_path = os.path.join(AUDIO_BASE, category, name + ".mp3")
        if os.path.exists(audio_path):
            return send_file(audio_path, mimetype="audio/mpeg")
    return jsonify({"error": "Audio not found"}), 404


@vocab_bp.route("/api/tts/google")
def tts_google():
    word = request.args.get("word", "")
    lang = request.args.get("lang", "en-GB")
    if not word:
        return jsonify({"error": "Missing word"}), 400
    if lang not in _ALLOWED_LANGS:
        return jsonify({"error": "Invalid lang"}), 400
    try:
        from flask import current_app
        return current_app.response_class(_fetch_google(word, lang), mimetype="audio/mpeg")
    except Exception as e:
        _fetch_google.cache_clear()
        return jsonify({"error": str(e)}), 500


@vocab_bp.route("/api/tts/baidu")
def tts_baidu():
    word = request.args.get("word", "")
    lang = request.args.get("lang", "en")
    if not word:
        return jsonify({"error": "Missing word"}), 400
    if lang not in _ALLOWED_LANGS:
        return jsonify({"error": "Invalid lang"}), 400
    try:
        from flask import current_app
        return current_app.response_class(_fetch_baidu(word, lang), mimetype="audio/mpeg")
    except Exception as e:
        _fetch_baidu.cache_clear()
        return jsonify({"error": str(e)}), 500


@vocab_bp.route("/api/tts/edge")
def tts_edge():
    word   = request.args.get("word", "")
    lang   = request.args.get("lang", "en-GB")
    gender = request.args.get("gender", "female")
    if not word:
        return jsonify({"error": "Missing word"}), 400
    if lang not in _ALLOWED_LANGS or gender not in ("male", "female"):
        return jsonify({"error": "Invalid lang or gender"}), 400
    voice = _EDGE_VOICES.get((lang, gender), "en-GB-SoniaNeural")
    try:
        from flask import current_app
        return current_app.response_class(_fetch_edge(word, voice), mimetype="audio/mpeg")
    except Exception as e:
        _fetch_edge.cache_clear()
        return jsonify({"error": str(e)}), 500
