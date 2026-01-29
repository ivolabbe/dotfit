import numpy as np

from dotfit.emission_lines import get_line_wavelengths, find_nearest_key, get_line_keys, single_line_dict


def test_get_line_wavelengths_and_types():
    lw, lr = get_line_wavelengths()
    assert isinstance(lw, dict)
    assert isinstance(lr, dict)
    assert len(lw) > 0
    k = next(iter(lw))
    assert isinstance(lw[k], list)
    assert all(isinstance(v, (float, np.floating, int, np.integer)) for v in lw[k])


def test_legacy_helpers_behave_consistently():
    lw, lr = get_line_wavelengths()
    k = next(iter(lw))
    w = lw[k][0]
    nearest = find_nearest_key(lw, w)
    assert isinstance(nearest, str)
    keys = get_line_keys(lw, k)
    assert isinstance(keys, list)
    singledict = single_line_dict(lw)
    assert isinstance(singledict, dict)
    assert nearest in lw or nearest in singledict
