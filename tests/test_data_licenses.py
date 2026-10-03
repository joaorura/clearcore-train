import pytest

from cctrain.data.licenses import TRAIN_ALLOWED, LicenseRefused, Role, check_train_allowed


@pytest.mark.parametrize("lic", sorted(TRAIN_ALLOWED))
def test_allowlisted_licenses_pass_for_train(lic):
    check_train_allowed(lic, Role.TRAIN, what="x")


def test_allowlist_is_exactly_the_spec_set_plus_generated():
    assert TRAIN_ALLOWED == {
        "CC0-1.0",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "Apache-2.0",
        "LicenseRef-public-domain-declared",
        "LicenseRef-cctrain-generated",
    }


@pytest.mark.parametrize(
    "lic",
    ["CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-SA-3.0", "CC-BY-ND-4.0", "ODbL-1.0", "DbCL-1.0",
     "GPL-3.0-or-later", "LicenseRef-research-only", "unknown", "", "cc-by-4.0"],
)
def test_everything_else_is_refused(lic):
    with pytest.raises(LicenseRefused):
        check_train_allowed(lic, Role.TRAIN, what="x")


def test_test_only_role_is_refused_even_with_permissive_license():
    with pytest.raises(LicenseRefused, match="test_only"):
        check_train_allowed("CC-BY-4.0", Role.TEST_ONLY, what="demand.wav")


def test_unknown_role_is_refused():
    with pytest.raises(ValueError):
        check_train_allowed("CC0-1.0", "whatever", what="x")
