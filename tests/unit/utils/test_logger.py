from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.utils.logger import ExperimentProgressLogger


@pytest.fixture
def mock_logger():
    with patch("dcv_benchmark.utils.logger.get_logger") as mock_get_logger:
        yield mock_get_logger.return_value


def test_start_log(mock_logger):
    progress_logger = ExperimentProgressLogger(total_samples=100)
    progress_logger.start()

    # helper to check if start message was logged
    mock_logger.info.assert_any_call(
        "🚀 [STARTED] Experiment started with 100 samples."
    )


def test_log_progress_logic(mock_logger):
    # Total samples 100 -> Interval 10
    progress_logger = ExperimentProgressLogger(total_samples=100)
    progress_logger.start()

    # Call with count 5 (should NOT log)
    progress_logger.log_progress(current_count=5, success_count=4)
    # Verify info was NOT called with progress message
    # We need to be careful not to match the start message
    # Check that no calls matching "Progress:" were made
    for call in mock_logger.info.call_args_list:
        assert "Progress:" not in call[0][0]

    # Call with count 10 (should log)
    progress_logger.log_progress(current_count=10, success_count=8)

    found_progress = False
    for call in mock_logger.info.call_args_list:
        if "Progress: 10/100" in call[0][0]:
            found_progress = True
            assert "Success Rate: 80.0%" in call[0][0]
            break
    assert found_progress, "Did not find expected progress log at 10%"


@patch("dcv_benchmark.utils.logger.datetime")
def test_eta_calculation(mock_datetime_module, mock_logger):
    # Setup start time
    start_time = MagicMock()
    now_time_2 = MagicMock()

    # We need now() to return start_time first, then a later time
    mock_datetime_module.datetime.now.side_effect = [start_time, now_time_2]

    # Setup elapsed time
    # elapsed = now - start
    mock_timedelta = MagicMock()
    mock_timedelta.total_seconds.return_value = 60.0  # 60 seconds elapsed

    # When (now - start) is called, return mock_timedelta
    now_time_2.__sub__.return_value = mock_timedelta

    progress_logger = ExperimentProgressLogger(total_samples=100)
    progress_logger.start()  # This calls now() -> start_time

    # Next call to log_progress calls now() -> now_time_2
    # Current count 10. Elapsed 60s.
    # Avg time = 6s/sample. Remaining 90. ETA = 540s = 9 mins.
    progress_logger.log_progress(current_count=10, success_count=10)

    found_eta = False
    for call in mock_logger.info.call_args_list:
        msg = call[0][0]
        if "ETA:" in msg:
            found_eta = True
            assert "~9 min" in msg
            break

    assert found_eta, "ETA was not logged or incorrect"


def test_last_sample_always_logged(mock_logger):
    progress_logger = ExperimentProgressLogger(total_samples=23)  # Interval 2
    progress_logger.start()

    # 23 % 2 != 0, so it wouldn't log by interval arithmetic usually (1..23)
    # But it IS the last sample, so it MUST log.
    progress_logger.log_progress(current_count=23, success_count=23)

    found_completion = False
    for call in mock_logger.info.call_args_list:
        if "Progress: 23/23" in call[0][0]:
            found_completion = True
            assert "100%" in call[0][0]

    assert found_completion
