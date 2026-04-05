import pandas as pd

from promo_uplift.ab_testing import difference_in_means


def test_difference_in_means_detects_positive_lift() -> None:
    df = pd.DataFrame(
        {
            "treatment": [0, 0, 0, 1, 1, 1],
            "conversion": [0, 0, 1, 1, 1, 1],
        }
    )
    result = difference_in_means(df, "conversion")
    assert result.ate > 0
    assert result.treatment_rate > result.control_rate

