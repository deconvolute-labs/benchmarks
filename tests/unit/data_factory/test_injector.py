import pytest

from dcv_benchmark.data_factory.injector import AttackInjector
from dcv_benchmark.models.data_factory import DataFactoryConfig


@pytest.fixture
def basic_config():
    """Returns a standard configuration for testing."""
    return DataFactoryConfig(
        dataset_name="test_ds",
        description="test",
        source_file="dummy.json",
        attack_strategy="naive",  # default
        attack_rate=1.0,
        truncate_overflow=False,
        flooding_repetitions=3,
        attack_payload="PAYLOAD",
        templates={
            "context_ignoring": "Start {payload} End",
            "fake_system": "Sys {payload} Sys",
        },
    )


@pytest.fixture
def injector(basic_config):
    return AttackInjector(basic_config)


def test_inject_naive(injector):
    """Test standard append behavior."""
    text = "Original Text"
    payload = "Attack"

    result = injector.inject(text, payload, strategy="naive")
    assert result == "Original Text\n\nAttack"


def test_inject_prefix(injector):
    """Test prepend behavior."""
    text = "Original Text"
    payload = "Attack"

    result = injector.inject(text, payload, strategy="prefix_injection")
    assert result == "Attack\n\nOriginal Text"


def test_inject_flooding(injector):
    """Test repetition logic."""
    text = "Original"
    payload = "Bad"
    # Config sets repetitions to 3
    result = injector.inject(text, payload, strategy="context_flooding")

    expected_payload = "Bad Bad Bad"
    assert result == f"Original\n\n{expected_payload}"


def test_inject_templates(injector):
    """Test that custom templates from config are used."""
    text = "Original"
    payload = "Malicious"

    # Test context_ignoring template: "Start {payload} End"
    result = injector.inject(text, payload, strategy="context_ignoring")
    assert result == "Original\n\nStart Malicious End"

    # Test fake_system template: "Sys {payload} Sys"
    result = injector.inject(text, payload, strategy="fake_system")
    assert result == "Original\n\nSys Malicious Sys"


def test_inject_leet_speak(injector):
    """Test character substitution."""
    text = "Original"
    payload = "leet"  # l -> l, e -> 3, t -> 7

    result = injector.inject(text, payload, strategy="leet_speak")
    # 'l' is not in map, 'e'->'3', 't'->'7'
    # map: a->@, e->3, i->1, o->0, s->$, t->7
    assert "l337" in result


def test_inject_fallback(injector):
    """Test that unknown strategies fall back to naive."""
    text = "Original"
    payload = "Attack"

    # "random_strategy" is not defined, should behave like naive
    result = injector.inject(text, payload, strategy="random_strategy")
    assert result == "Original\n\nAttack"


def test_truncation_logic():
    """Test that text is truncated to fit payload when enabled."""
    config = DataFactoryConfig(
        dataset_name="test",
        description="test",
        source_file="dummy",
        attack_strategy="naive",
        truncate_overflow=True,  # ENABLED
        attack_payload="PAYLOAD",
    )
    injector = AttackInjector(config)

    # Original text is long: 50 chars
    long_text = "x" * 50
    payload = "ATTACK"

    # Combined length without truncation would be 50 + 2 + 6 = 58
    # With truncation, we expect total length <= 50 (approx, logic cuts text)

    result = injector.inject(long_text, payload, strategy="naive")

    # Payload must always be present in full
    assert payload in result

    # Original text should be shortened
    assert len(result) < 58
    assert "..." in result


def test_truncation_payload_too_big():
    """Test edge case where payload > text."""
    config = DataFactoryConfig(
        dataset_name="test",
        description="test",
        source_file="dummy",
        attack_strategy="naive",
        truncate_overflow=True,
        attack_payload="PAYLOAD",
    )
    injector = AttackInjector(config)

    short_text = "Hi"
    payload = "VERY_LONG_PAYLOAD"

    result = injector.inject(short_text, payload, strategy="naive")

    # Should replace entirely because it can't fit both
    assert result == payload
