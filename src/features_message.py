import re

URGENCY_WORDS = [
    "urgent", "immediately", "asap", "today", "now",
    "right away", "as soon as possible"
]

AUTHORITY_WORDS = [
    "ceo", "cfo", "director", "president", "vp", "vice president",
    "manager", "executive"
]

PAYMENT_WORDS = [
    "wire", "transfer", "payment", "bank account", "account details",
    "vendor", "supplier", "invoice", "remit", "funds"
]

SECRECY_WORDS = [
    "confidential", "do not call", "don't call", "secret",
    "privately", "discreet", "keep this between us"
]


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    text = text.lower()
    return sum(1 for word in keywords if word in text)


def has_money_amount(text: str) -> int:
    return int(bool(re.search(r"\$?\d{3,}", text)))


def has_same_day_pressure(text: str) -> int:
    text = text.lower()
    phrases = ["today", "now", "immediately", "asap", "right away"]
    return int(any(p in text for p in phrases))


def has_bank_change_language(text: str) -> int:
    text = text.lower()
    phrases = [
        "updated account",
        "new account",
        "change bank details",
        "update bank details",
        "update vendor bank",
        "new banking instructions"
    ]
    return int(any(p in text for p in phrases))


def extract_manual_features(text: str) -> dict:
    return {
        "urgency_score": count_keyword_hits(text, URGENCY_WORDS),
        "authority_score": count_keyword_hits(text, AUTHORITY_WORDS),
        "payment_score": count_keyword_hits(text, PAYMENT_WORDS),
        "secrecy_score": count_keyword_hits(text, SECRECY_WORDS),
        "amount_flag": has_money_amount(text),
        "same_day_flag": has_same_day_pressure(text),
        "bank_change_flag": has_bank_change_language(text),
        "char_len": len(text),
        "word_count": len(text.split()),
        "exclamation_count": text.count("!"),
    }
