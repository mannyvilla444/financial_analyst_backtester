"""
telegram_share.py

Send an HTML report to a Telegram channel or group as a document, using the
same bot token as the Telegram bot (TELEGRAM_BOT_TOKEN in .env).

The destination is given as a public link and turned into a Bot API chat_id:
    https://t.me/mychannel   t.me/mychannel   t.me/s/mychannel
    @mychannel               mychannel        → "@mychannel"
    -1001234567890                            → numeric id (private channels)
    123456789                                 → your personal chat ID (private chat with the bot)
    https://t.me/<your_bot>                   → your private chat, via TELEGRAM_CHAT_ID in .env

Requirements on the Telegram side:
  • channel → the bot must be added as an administrator with "Post messages"
  • group   → the bot must be a member

Previously used destinations are cached in telegram_links.json (most recent
first) so the UI can offer them again.
"""

import json
import logging
import os
import re
import threading
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/{method}"
MAX_LINKS = 25
_lock = threading.Lock()
_bot_info_cache: dict = {}

USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{3,31}$")


class TelegramError(Exception):
    """User-facing error (message is safe to show in the UI)."""


def get_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()


# ── Link parsing ──────────────────────────────────────────────────────────────

def parse_link(link: str) -> tuple[str, str]:
    """
    Public Telegram link → (chat_id, canonical display link).
    Raises TelegramError with a helpful message for private invite links etc.
    """
    s = (link or "").strip()
    if not s:
        raise TelegramError("Enter a Telegram channel link, e.g. https://t.me/yourchannel")

    if re.fullmatch(r"-?\d{5,20}", s):                     # numeric chat id
        return s, s

    s = re.sub(r"^(https?://)?(www\.)?", "", s, flags=re.I)
    s = re.sub(r"^(t\.me|telegram\.me|telegram\.dog)/", "", s, flags=re.I)
    if s.lower().startswith("s/"):                          # t.me/s/<name> web preview
        s = s[2:]
    if s.startswith("+") or s.lower().startswith("joinchat"):
        raise TelegramError(
            "That's a private invite link — bots can't post using those. "
            "Use the channel's public @username, or its numeric ID (starts with -100)."
        )
    s = s.lstrip("@").split("?")[0].split("/")[0]
    if not USERNAME_RE.match(s):
        raise TelegramError(
            f"“{link}” doesn't look like a public Telegram link. "
            "Use https://t.me/yourchannel or @yourchannel."
        )
    return f"@{s}", f"https://t.me/{s}"


# ── Cached links ──────────────────────────────────────────────────────────────

def load_links(path: str) -> list[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        return [d for d in data if isinstance(d, dict) and d.get("link")] if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning(f"telegram_links.json unreadable ({e}); ignoring")
        return []


def _save_links(path: str, links: list[dict]):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(links, f, indent=1)
    os.replace(tmp, path)


def remember_link(path: str, link: str, title: str | None = None) -> list[dict]:
    with _lock:
        links = [d for d in load_links(path) if d["link"].lower() != link.lower()]
        links.insert(0, {"link": link, "title": title or "",
                         "last_used": datetime.now().isoformat(timespec="seconds")})
        links = links[:MAX_LINKS]
        _save_links(path, links)
        return links


def forget_link(path: str, link: str) -> list[dict]:
    with _lock:
        links = [d for d in load_links(path) if d["link"].lower() != (link or "").lower()]
        _save_links(path, links)
        return links


# ── Bot API calls ─────────────────────────────────────────────────────────────

def _explain(description: str, chat_id: str, bot_username: str | None) -> str:
    d = (description or "").lower()
    bot = f"@{bot_username}" if bot_username else "your bot"
    is_user_id = bool(re.fullmatch(r"\d{5,20}", chat_id or ""))
    if "bots can't send messages to bots" in d or "bot can't send messages to bots" in d:
        return (f"{chat_id} is a bot, and bots can't message other bots. Use a channel link, "
                f"or your own chat ID to get reports in your private chat with {bot}.")
    if "can't initiate conversation" in d or "bot was blocked" in d or (is_user_id and "chat not found" in d):
        return (f"{bot} can't message you yet. Open {bot} in Telegram and press Start "
                f"(or unblock it), then send again.")
    if "chat not found" in d:
        return (f"Telegram can't find {chat_id}. Check the link is the channel's public "
                f"username, and that {bot} has been added to it.")
    if "not a member" in d or "have no rights" in d or "not enough rights" in d or "forbidden" in d:
        return (f"{bot} isn't allowed to post in {chat_id}. Add it to the channel as an "
                f"administrator with “Post messages” permission, then try again.")
    if "unauthorized" in d:
        return "Telegram rejected the bot token. Check TELEGRAM_BOT_TOKEN in your .env file."
    if "too big" in d or "entity too large" in d:
        return "The report is too large for Telegram."
    return f"Telegram error: {description}"


def bot_info(token: str | None = None) -> dict | None:
    """getMe → {'username': ..., 'name': ...} (cached per token)."""
    token = token or get_token()
    if not token:
        return None
    if token in _bot_info_cache:
        return _bot_info_cache[token]
    try:
        r = requests.get(API.format(token=token, method="getMe"), timeout=10)
        data = r.json()
        if data.get("ok"):
            info = {"username": data["result"].get("username"),
                    "name": data["result"].get("first_name"),
                    "id": data["result"].get("id")}
            _bot_info_cache[token] = info
            return info
        return {"error": _explain(data.get("description", ""), "", None)}
    except Exception as e:
        return {"error": f"Couldn't reach Telegram: {e}"}


def _resolve_own_bot(chat_id: str, info: dict) -> str:
    """
    The bot's own link (t.me/<your_bot>) means "send to my private chat with the
    bot". A bot can't post to itself, so route it to TELEGRAM_CHAT_ID.
    """
    bot_user = (info.get("username") or "").lower()
    if bot_user and chat_id.lower() == f"@{bot_user}":
        own_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not re.fullmatch(r"-?\d{5,20}", own_chat):
            raise TelegramError(
                f"That's your bot's own link. A bot can't post to itself. To get reports in your "
                f"private chat with @{info['username']}: open that chat in Telegram, press Start, "
                f"and send /chatid (your Telegram bot replies with your chat ID). Then either "
                f"paste that number here, or add TELEGRAM_CHAT_ID=<number> to .env and restart "
                f"so this link works. To post to a channel instead, use the channel's link."
            )
        return own_chat
    return chat_id


def send_document(link: str, filename: str, content: bytes, caption: str = "",
                  token: str | None = None) -> dict:
    """
    Send `content` as a document to the chat behind `link`.
    Returns {"chat_id", "link", "chat_title", "message_id"}; raises TelegramError.
    """
    token = token or get_token()
    if not token:
        raise TelegramError(
            "Telegram isn't set up: add TELEGRAM_BOT_TOKEN=<your bot token> to the .env "
            "file in the finbot-web folder (the same token your Telegram bot uses), "
            "then restart the app."
        )
    chat_id, canonical = parse_link(link)
    info = bot_info(token) or {}

    chat_id = _resolve_own_bot(chat_id, info)

    try:
        r = requests.post(
            API.format(token=token, method="sendDocument"),
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"document": (filename, content, "text/html")},
            timeout=60,
        )
        data = r.json()
    except requests.RequestException as e:
        raise TelegramError(f"Couldn't reach Telegram: {e}")
    except ValueError:
        raise TelegramError(f"Unexpected response from Telegram (HTTP {r.status_code}).")

    if not data.get("ok"):
        raise TelegramError(_explain(data.get("description", ""), chat_id, info.get("username")))

    chat = data["result"].get("chat", {})
    return {
        "chat_id":    chat_id,
        "link":       canonical,
        "chat_title": (chat.get("title")
                       or (f"your chat with @{info['username']}" if chat.get("type") == "private" and info.get("username")
                           else chat.get("first_name") or chat.get("username") or chat_id)),
        "message_id": data["result"].get("message_id"),
    }


# ── Connection test ───────────────────────────────────────────────────────────

def _api(token: str, method: str, **params) -> dict:
    try:
        r = requests.get(API.format(token=token, method=method), params=params, timeout=15)
        return r.json()
    except Exception as e:
        return {"ok": False, "description": f"Couldn't reach Telegram: {e}"}


def diagnose(link: str, token: str | None = None) -> dict:
    """
    Step-by-step check of everything a send needs, without posting anything.
    Returns {"ok": bool, "verdict": str, "checks": [{"ok": True|False|None, "label", "detail"}]}.
    """
    checks = []

    def add(ok, label, detail=""):
        checks.append({"ok": ok, "label": label, "detail": detail})

    def done(verdict):
        return {"ok": all(c["ok"] is not False for c in checks), "verdict": verdict, "checks": checks}

    token = token or get_token()
    if not token:
        add(False, "Bot token", "No TELEGRAM_BOT_TOKEN was loaded.")
        return done("Add TELEGRAM_BOT_TOKEN to the .env file in the app folder shown below, then restart the app.")

    me = _api(token, "getMe")
    if not me.get("ok"):
        add(False, "Bot token", me.get("description", "Rejected by Telegram"))
        return done("Telegram rejected the token. Copy it again from @BotFather (or your bot's .env), "
                    "with no spaces or quotes, then restart the app.")
    bot = me["result"]
    info = {"username": bot.get("username"), "id": bot.get("id")}
    add(True, "Bot token", f"Valid: the bot is @{bot.get('username')}")

    try:
        chat_id, _ = parse_link(link)
        chat_id = _resolve_own_bot(chat_id, info)
    except TelegramError as e:
        add(False, "Destination", str(e))
        return done("Fix the destination and test again.")
    add(True, "Destination", f"Link understood as {chat_id}")

    chat = _api(token, "getChat", chat_id=chat_id)
    if not chat.get("ok"):
        desc = chat.get("description", "")
        if re.fullmatch(r"\d{5,20}", chat_id):
            add(False, "Telegram can see the chat", desc)
            return done(f"Open @{bot.get('username')} in Telegram and press Start, then test again. "
                        f"Also check that the number is your chat ID.")
        add(False, "Telegram can see the chat", desc)
        return done(
            f"Telegram can't find {chat_id}. Either (1) the username is misspelled, (2) the channel is "
            f"private — it has no public t.me/<name> link, so enter its numeric ID starting with -100 instead "
            f"(forward any post from the channel to @userinfobot to get it), or (3) the bot hasn't been "
            f"added to it yet."
        )
    c = chat["result"]
    kind = c.get("type")
    name = c.get("title") or c.get("first_name") or c.get("username") or chat_id
    add(True, "Telegram can see the chat", f"{name} ({kind})")

    if kind == "private":
        return done(f"Ready: reports will go to your private chat with @{bot.get('username')}.")

    member = _api(token, "getChatMember", chat_id=chat_id, user_id=bot.get("id"))
    if not member.get("ok"):
        add(False, "Bot is in the chat", member.get("description", ""))
        return done(f"Add @{bot.get('username')} to “{name}” as an administrator "
                    f"(channel → Administrators → Add Admin), allow “Post messages”, then test again.")
    status = member["result"].get("status")
    if status in ("left", "kicked"):
        add(False, "Bot is in the chat", f"status: {status}")
        return done(f"@{bot.get('username')} isn't in “{name}”. Add it as an administrator with "
                    f"“Post messages” permission, then test again.")
    if kind == "channel":
        if status == "creator":
            add(True, "Bot can post", "owner")
        elif status == "administrator":
            can_post = member["result"].get("can_post_messages")
            if can_post is False:
                add(False, "Bot can post", "admin, but “Post messages” is switched off")
                return done(f"In “{name}” → Administrators → @{bot.get('username')}, switch on "
                            f"“Post messages” (sometimes shown as “Post Messages” under “Manage messages”).")
            add(True, "Bot can post", "admin with “Post messages”")
        else:
            add(False, "Bot can post", f"status: {status} (subscribers can't post in channels)")
            return done(f"@{bot.get('username')} is only a subscriber of “{name}”. Promote it: "
                        f"channel → Administrators → Add Admin → @{bot.get('username')} → allow “Post messages”.")
    else:
        if status == "restricted" and member["result"].get("can_send_messages") is False:
            add(False, "Bot can post", "restricted from sending messages")
            return done(f"Remove the restriction on @{bot.get('username')} in “{name}”.")
        add(True, "Bot can post", f"status: {status}")

    return done(f"Everything checks out. Sending to “{name}” should work now.")
