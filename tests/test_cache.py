"""Answer cache: partitions, typo/meaning matching, safety guards."""
from app.cache import AnswerCache, make_partition
from tests.test_chat_api import post

BASE = dict(role="employee", region="IN", today="2026-10-12",
            ref_date="2026-10-12", holiday=False,
            corpus_hash="abc", version="v1")


def part(**change):
    return make_partition(**{**BASE, **change})


def new_cache(**kw):
    return AnswerCache(enabled=True, **kw)


def test_partition_changes_with_every_part():
    p = part()
    for change in [dict(role="manager"), dict(role="hr"),
                   dict(region="DE"), dict(today="2026-10-13"),
                   dict(ref_date="2025-12-31"), dict(holiday=True),
                   dict(corpus_hash="xyz"), dict(version="v2")]:
        assert part(**change) != p


def test_exact_and_spelling_mistakes_hit():
    c = new_cache()
    c.store(part(), "How many sick days do I get?", "A")
    for q in ["how many SICK days do i get",
              "How many sick dyas do I get?",
              "How mny sik days do i get",
              "how many sick days i get"]:
        assert c.lookup(part(), q)[0] == "A", q


def test_different_questions_miss():
    c = new_cache()
    c.store(part(), "How many sick days do I get?", "A")
    for q in ["How many annual days do I get?",
              "Do sick days carry over?",
              "How many sick days do I not get?"]:
        assert c.lookup(part(), q)[0] is None, q


def test_numbers_must_match():
    c = new_cache()
    c.store(part(), "Can I carry forward 5 days?", "B")
    assert c.lookup(part(), "Can I carry forward 20 days?")[0] is None
    assert c.lookup(part(), "Can I cary forward 5 days?")[0] == "B"


def test_other_partition_never_hits():
    c = new_cache()
    c.store(part(), "How many sick days do I get?", "A")
    assert c.lookup(part(role="hr"), "How many sick days do I get?")[0] is None
    assert c.lookup(part(region="DE"), "How many sick days do I get?")[0] is None


def test_meaning_match_with_embeddings():
    vecs = {"how many sick days do i get": [1.0, 0.0],
            "what is my sick leave allowance": [0.99, 0.05],
            "what is the travel policy": [0.0, 1.0]}
    c = new_cache(embed_fn=lambda t: vecs.get(t, [0.5, 0.5]))
    c.store(part(), "How many sick days do I get?", "A")
    assert c.lookup(part(), "What is my sick leave allowance?")[0] == "A"
    assert c.lookup(part(), "What is the travel policy?")[0] is None


def test_expiry_size_and_disable():
    c = new_cache(ttl_s=-1)
    c.store(part(), "q one", 1)
    assert c.lookup(part(), "q one")[0] is None
    c = new_cache(max_items=2)
    for q in ["alpha question", "bravo question", "charlie question"]:
        c.store(part(), q, q)
    assert c.lookup(part(), "alpha question")[0] is None
    c = AnswerCache(enabled=False)
    c.store(part(), "q", 1)
    assert c.lookup(part(), "q")[0] is None
def test_key_words_must_match():
    pairs = [
        ("What are the public holidays in June?",
         "What are the public holidays in July?"),
        ("What are the public holidays in March?",
         "What are the public holidays in May?"),
        ("What is the maternity leave?", "What is the paternity leave?"),
        ("How many sick days do I get?", "How many sick days did I get?"),
        ("Can I work from home on Friday?",
         "Can I work from home on Monday?"),
        ("What is the hotel limit per night?",
         "What is the hotel limit per day?"),
        ("How much annual leave do I get?",
         "How much sick leave do I get?"),
    ]
    for cached, asked in pairs:
        c = new_cache()
        c.store(part(), cached, "X")
        assert c.lookup(part(), asked)[0] is None, asked


def test_typos_in_key_words_still_hit():
    c = new_cache()
    c.store(part(), "What is the maternity leave?", "M")
    assert c.lookup(part(), "What is the matrnity leave?")[0] == "M"
    c.store(part(), "What are the public holidays in June?", "J")
    assert c.lookup(part(), "What are the public holidays in Jnue?")[0] == "J"

# ---------------- integration through POST /chat ----------------

def answer_calls(fake_llm):
    return sum(1 for c in fake_llm.calls if c["purpose"] == "answer")


def test_typo_hits_cache_through_chat(client, fake_llm):
    r1 = post(client, "E1002", "k1", "How many sick days do I get?").json()
    n = answer_calls(fake_llm)
    r2 = post(client, "E1002", "k2", "How many sick dyas do I get?").json()
    assert answer_calls(fake_llm) == n
    assert r2["route"] == "rag" and r2["citations"] == r1["citations"]


def test_other_role_or_region_misses(client, fake_llm):
    post(client, "E1002", "k3", "How many sick days do I get?")
    n = answer_calls(fake_llm)
    post(client, "E1001", "k4", "How many sick days do I get?")  # manager
    post(client, "E1003", "k5", "How many sick days do I get?")  # DE
    assert answer_calls(fake_llm) == n + 2


def test_hr_answer_never_served_to_employee(client):
    post(client, "E1000", "k6", "What is the salary band for L3?")
    r = post(client, "E1002", "k7", "What is the salary band for L3?").json()
    assert all(c["doc_id"] != "HR-COMP" for c in r["citations"])


def test_tool_answers_not_cached(client):
    a = post(client, "E1002", "k8", "what is my balance").json()
    b = post(client, "E1002", "k9", "what is my balance").json()
    assert a["route"] == b["route"] == "tool"
    assert b["tool_calls"][0]["status"] == "success"