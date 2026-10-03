import json
from pathlib import Path

import numpy as np
import pytest
from enroll_cases import cases

from cctrain.data import enroll
from cctrain.data.enroll import EnrollmentRejected, check_enrollment_audio, validate_enrollment_audio

GOLDEN = json.loads((Path(__file__).parent / "golden" / "enroll_validation_v1.json").read_text())["cases"]
CASES = cases()


def test_golden_covers_every_case():
    assert set(GOLDEN) == set(CASES)


def test_constants_mirror_the_rust_contract():
    assert (enroll.MIN_SAMPLES, enroll.MAX_SAMPLES) == (96_000, 192_000)
    assert enroll.MAX_PEAK == np.float32(0.99)
    assert enroll.MIN_RMS_DBFS == np.float32(-40.0)
    assert enroll.MIN_ACTIVE_FRACTION == np.float32(0.60)
    assert (enroll.VAD_FRAME_SAMPLES, enroll.VAD_ABSOLUTE_FLOOR_DBFS, enroll.VAD_RELATIVE_RANGE_DB) == (320, -50.0, 30.0)


@pytest.mark.parametrize("name", sorted(CASES))
def test_decision_and_measurements_match_rust(name):
    want = GOLDEN[name]
    x = CASES[name]
    if want["status"] == "ok":
        validate_enrollment_audio(x)
        report = check_enrollment_audio(x)
        assert report.peak == np.float32(want["peak"])
        assert report.rms_dbfs == np.float32(want["rms_dbfs"])
        assert report.active_speech_fraction == np.float32(want["active_fraction"])
        assert report.duration_secs == np.float32(len(x)) / np.float32(16_000)
    else:
        with pytest.raises(EnrollmentRejected) as err:
            validate_enrollment_audio(x)
        assert err.value.reason == want["status"]
        assert np.float32(err.value.value) == np.float32(want["value"])


def test_rejection_is_a_value_error_with_a_readable_message():
    with pytest.raises(ValueError, match="too_short"):
        validate_enrollment_audio(np.zeros(10, np.float32))


def test_input_dtype_and_shape_contract():
    with pytest.raises(ValueError, match="1-D"):
        validate_enrollment_audio(np.zeros((2, 96_000), np.float32))
    x = np.full(96_000, 0.1, np.float64)  # f64 é convertido para f32 como no Rust (f32 na entrada)
    validate_enrollment_audio(x)


def test_validation_does_not_mutate_the_input():
    x = CASES["ok_dc_6s"].copy()
    validate_enrollment_audio(x)
    assert np.array_equal(x, CASES["ok_dc_6s"])
