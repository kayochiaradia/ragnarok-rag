"""Normalização de texto e expansão de sinônimos do domínio Ragnarok Online.

Esta é a parte que faz um RAG de domínio funcionar de verdade: o jogador
escreve "gtb", "sg", "asura", "pt", "upar" — e o corpus usa os nomes por
extenso. Sem essa ponte, a busca lexical falha e a vetorial fica no limite.
"""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "a", "à", "ao", "aos", "as", "às", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "essa", "esse", "esta", "este", "eu", "é", "faz", "fazer",
    "isso", "já", "mais", "mas", "me", "melhor", "meu", "minha", "na", "nas",
    "no", "nos", "num", "o", "os", "ou", "para", "pra", "por", "qual", "quais",
    "quando", "que", "quem", "se", "sem", "ser", "seu", "sua", "são", "só",
    "tem", "ter", "um", "uma", "vc", "você", "onde", "oque", "sobre", "the",
    "of", "and", "is", "what", "how",
}

# Sigla ou gíria -> termos canônicos usados no corpus.
SYNONYMS: dict[str, list[str]] = {
    "gtb": ["golden", "thief", "bug", "card", "imunidade", "magia"],
    "gr": ["ghostring", "card", "fantasma"],
    "thara": ["thara", "frog", "demi-humano"],
    "ray": ["raydric", "card", "neutro"],
    "ygg": ["yggdrasil", "berry", "seed"],
    "ori": ["oridecon"],
    "elu": ["elunium"],
    "oca": ["old", "card", "album"],
    "obb": ["old", "blue", "box"],
    "ovb": ["old", "violet", "box"],
    "db": ["dead", "branch"],
    "sg": ["storm", "gust"],
    "lov": ["lord", "of", "vermilion"],
    "ms": ["meteor", "storm"],
    "jt": ["jupitel", "thunder"],
    "fw": ["fire", "wall"],
    "sw": ["safety", "wall"],
    "lp": ["land", "protector"],
    "me": ["magnus", "exorcismus"],
    "bb": ["bowling", "bash", "bloody", "branch"],
    "ds": ["double", "strafe"],
    "sb": ["sonic", "blow"],
    "sbk": ["soul", "breaker"],
    "ad": ["acid", "demonstration"],
    "edp": ["enchant", "deadly", "poison"],
    "asura": ["asura", "strike", "guillotine", "fist", "mestre", "champion"],
    "fs": ["full", "support", "sacerdote", "priest"],
    "bp": ["battle", "priest", "sacerdote"],
    "lk": ["lord", "knight", "lorde", "cavaleiro"],
    "hw": ["high", "wizard", "arquimago"],
    "hp": ["high", "priest", "sumo", "sacerdote", "vida"],
    "ws": ["whitesmith", "mestre", "ferreiro"],
    "ac": ["assassin", "cross", "algoz"],
    "sinx": ["assassin", "cross", "algoz"],
    "gc": ["guillotine", "cross", "sicario"],
    "sn": ["super", "novice", "aprendiz"],
    "prof": ["professor"],
    "champ": ["champion", "mestre"],
    "creo": ["creator", "criador"],
    "woe": ["guerra", "do", "emperium", "war"],
    "emp": ["emperium"],
    "pvp": ["pvp", "jogador", "contra", "jogador"],
    "pt": ["party", "grupo"],
    "ks": ["kill", "steal", "roubar", "monstro"],
    "mvp": ["mvp", "boss", "chefe"],
    "aspd": ["aspd", "velocidade", "de", "ataque"],
    "atk": ["atk", "ataque"],
    "matk": ["matk", "ataque", "magico"],
    "def": ["def", "defesa"],
    "mdef": ["mdef", "defesa", "magica"],
    "exp": ["experiencia", "exp"],
    "npc": ["npc"],
    "re": ["renewal"],
    "pre": ["pre-renewal", "prerenewal", "classico"],
    "upar": ["upar", "leveling", "nivel", "experiencia"],
    "up": ["upar", "leveling", "nivel"],
    "farmar": ["farm", "farmar", "cacar", "zeny"],
    "buffar": ["buff", "suporte", "blessing", "agi"],
    "tankar": ["tanque", "tank", "vit", "defesa"],
    "ninar": ["sleep", "dormir", "estado"],
    "card": ["carta", "card"],
    "cards": ["cartas", "card"],
    "carta": ["carta", "card"],
    "job": ["job", "classe", "profissao"],
    "classe": ["classe", "job", "profissao"],
    "grana": ["zeny", "dinheiro", "economia"],
    "dinheiro": ["zeny", "dinheiro", "economia"],
    "money": ["zeny", "dinheiro"],
    "chapeu": ["headgear", "capacete", "chapeu"],
    "arma": ["arma", "weapon", "atk"],
    "refinar": ["refino", "refinar", "upgrade"],
    "quebrar": ["quebrar", "refino", "falha"],
    "mapa": ["mapa", "cidade", "masmorra"],
    "dungeon": ["masmorra", "dungeon"],
    "boss": ["mvp", "chefe", "boss"],
    "elemento": ["elemento", "elemental", "propriedade"],
    "mob": ["monstro", "mob"],
    "spot": ["spot", "local", "cacada", "leveling"],
    "build": ["build", "status", "distribuicao"],
    "status": ["status", "atributo", "build"],
    "skill": ["habilidade", "skill", "magia"],
    "skills": ["habilidades", "skill", "magia"],
    "pet": ["pet", "mascote"],
    "homun": ["homunculus", "homunculo"],
    "homunculo": ["homunculus", "homunculo", "alquimista"],
    "merc": ["mercenario"],
    "peco": ["peco", "montaria"],
    "pegar": ["obter", "invocar", "invocado", "call"],
    "conseguir": ["obter", "invocar"],
    "quanto": ["percentual", "porcentagem", "multiplicador", "cento"],
}


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    """Minúsculas, sem acento, pontuação virando espaço."""
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singularize(token: str) -> str:
    """Redução de plural mínima para português.

    Não é um stemmer completo de propósito: derrubar apenas o "s" final resolve
    o grosso dos casos do domínio (cartas/carta, limites/limite, elementos/
    elemento, monstros/monstro) sem os falsos positivos que um Porter agressivo
    produziria em nomes próprios como "Poporing" ou "Byalan".
    """
    # ``status``, ``bonus`` e o nome inglês ``homunculus`` não são plurais.
    # Terminações ``is`` e ``ns`` também exigem regras próprias (animais,
    # itens); é mais seguro preservá-las do que produzir radicais incorretos.
    protected_endings = ("ss", "us", "is", "ns")
    if len(token) > 3 and token.endswith("s") and not token.endswith(protected_endings):
        return token[:-1]
    return token


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = normalize(text).split()
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return [singularize(t) for t in tokens]


def expand_query(query: str) -> list[str]:
    """Tokens da pergunta mais os sinônimos de domínio."""
    tokens = tokenize(query)
    expanded = list(tokens)
    for token in tokens:
        for extra in SYNONYMS.get(token, []):
            norm = normalize(extra)
            if norm and norm not in expanded:
                expanded.append(norm)
    return expanded


def char_ngrams(text: str, n: int = 4) -> list[str]:
    """N-gramas de caractere, usados pelo embedding de hashing."""
    norm = f" {normalize(text)} "
    if len(norm) <= n:
        return [norm]
    return [norm[i: i + n] for i in range(len(norm) - n + 1)]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?:])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]
