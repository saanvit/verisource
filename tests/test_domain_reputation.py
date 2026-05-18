from backend.reliability.domain_reputation import lookup


def test_curated_domain():
    rep = lookup("https://www.reuters.com/world/article")
    assert rep.domain == "reuters.com"
    assert rep.score >= 85
    assert rep.type == "wire"


def test_subdomain_falls_back_to_registered():
    rep = lookup("https://news.bbc.co.uk/something")
    assert rep.domain == "bbc.co.uk"
    assert rep.score >= 80


def test_tld_prior_for_gov():
    rep = lookup("https://example-agency.gov/page")
    assert rep.type == "gov"
    assert rep.score >= 80


def test_unknown_domain_neutral():
    rep = lookup("https://totally-made-up-news-site-xyz.example/article")
    assert 40 <= rep.score <= 60


def test_no_url_returns_neutral():
    rep = lookup(None)
    assert rep.domain == "unknown"
    assert rep.score == 50.0


def test_satire_domain_recognized():
    rep = lookup("https://theonion.com/doctors-confirm-rudy-giuliani-in-liquid-but-stable-condition/")
    assert rep.domain == "theonion.com"
    assert rep.type == "satire"
    assert rep.score <= 10
    assert "satire" in rep.rationale.lower()
