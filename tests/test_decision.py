from product_inspection import decide


def test_no_detection_is_pass():
    assert decide(None, 0.60, 0.15) == "Pass"


def test_low_confidence_is_pass():
    assert decide(0.10, 0.60, 0.15) == "Pass"


def test_middle_confidence_needs_review():
    assert decide(0.40, 0.60, 0.15) == "Review Required"


def test_high_confidence_is_fail():
    assert decide(0.80, 0.60, 0.15) == "Fail"
