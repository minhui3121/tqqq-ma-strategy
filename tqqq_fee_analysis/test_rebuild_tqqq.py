import math
import unittest


class RebuildTqqqTest(unittest.TestCase):
    def test_2010_02_12_row_math(self) -> None:
        base_tqqq_close = 0.21627600491046906
        base_ndx_close = 1775.739990234375
        next_ndx_close = 1779.1099853515625
        daily_drag = 9.266730443378356e-05

        ndx_return = next_ndx_close / base_ndx_close - 1.0
        expected_factor = 1.0 + (3.0 * ndx_return) - daily_drag
        expected_rebuilt_tqqq = base_tqqq_close * expected_factor

        expected_row = {
            "date": "2010-02-12",
            "ndx_close": next_ndx_close,
            "ndx_return": ndx_return,
            "factor": expected_factor,
            "rebuilt_tqqq": expected_rebuilt_tqqq,
            "actual_tqqq_close": 0.2171609997749328,
            "daily_drag": daily_drag,
            "diff_rebuilt_minus_actual": expected_rebuilt_tqqq - 0.2171609997749328,
        }

        self.assertEqual(expected_row["date"], "2010-02-12")
        self.assertTrue(math.isclose(expected_row["ndx_close"], 1779.1099853515625, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(expected_row["ndx_return"], 0.0018977976143583764, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(math.isclose(expected_row["factor"], 1.0056007255386414, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(math.isclose(expected_row["rebuilt_tqqq"], 0.2174873074545664, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(math.isclose(expected_row["actual_tqqq_close"], 0.2171609997749328, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(math.isclose(expected_row["daily_drag"], 9.266730443378356e-05, rel_tol=0.0, abs_tol=1e-18))
        self.assertTrue(math.isclose(expected_row["diff_rebuilt_minus_actual"], 0.00032630767963358847, rel_tol=0.0, abs_tol=1e-15))


if __name__ == "__main__":
    unittest.main()
