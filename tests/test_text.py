from ragnarok_rag.text import expand_query, tokenize


def test_plural_normalization_matches_singular_corpus_terms() -> None:
    assert tokenize("limites cartas elementos monstros") == [
        "limite",
        "carta",
        "elemento",
        "monstro",
    ]


def test_singularization_preserves_words_ending_in_us() -> None:
    assert tokenize("status homunculus") == ["status", "homunculus"]


def test_acquisition_question_expands_to_obtain_and_invoke_vocabulary() -> None:
    expanded = set(expand_query("como pegar homunculo"))

    assert {"obter", "invocar"} <= expanded
